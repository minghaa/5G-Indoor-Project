"""
Indoor Hotspot Simulator V2 for Multi-Cell 5G Environment
Based on DataDescription.xlsx 

Features:
- 12 Small Cells in 2x6 grid (ISD=20m)
- 120 UEs with pre-generated Random Walk mobility
- State with 173 dimensions (17 sim + 12 network + 12×12 cell)
- Power model: Sleep=15W, Active=50W base
- Channel model: 3GPP Indoor Office LOS with shadowing σ=4dB
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class CellState:
    """State of a single cell - matches DataDescription Cell Features"""
    id: int
    x: float
    y: float
    z: float
    
    # Cell Features (12 per cell)
    cpuUsage: float = 0.0           # %
    prbUsage: float = 0.0           # %
    currentLoad: float = 0.0        # load units
    maxCapacity: float = 250.0      # load units
    numConnectedUEs: int = 0        # count
    txPower: float = 23.0           # dBm
    energyConsumption: float = 0.0  # Watt
    avgRSRP: float = -80.0          # dBm
    avgRSRQ: float = -10.0          # dB
    avgSINR: float = 10.0           # dB
    totalTrafficDemand: float = 0.0 # traffic units
    loadRatio: float = 0.0          # 0-1
    
    # Internal state
    is_sleeping: bool = False
    wake_up_delay: int = 0
    power_ratio: float = 1.0        # Action value [0,1]


@dataclass
class UEState:
    """State of a User Equipment"""
    id: int
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    serving_cell: int = -1
    rsrp: float = -100.0  # dBm
    rsrq: float = -15.0   # dB
    sinr: float = 0.0     # dB
    is_dropped: bool = False
    latency: float = 0.0  # ms
    traffic_demand: float = 1.0


class IndoorHotspotSimulatorV2:
    """
    Simulator matching mentor's DataDescription specifications
    
    State Space (173 dimensions):
        - Simulation Features (17)
        - Network Features (12)
        - Cell Features (12 cells × 12 features = 144)
    """
    
    # === Simulation Config (from DataDescription) ===
    TOTAL_CELLS = 12
    TOTAL_UES = 120
    SIM_TIME = 300          # total simulation time in seconds
    TIME_STEP = 1.0         # seconds per step
    CARRIER_FREQUENCY = 4.0  # GHz (Mid-band 5G)
    ISD = 20.0              # Inter-Site Distance (meters)
    
    # Power parameters
    MIN_TX_POWER = 10.0     # dBm
    MAX_TX_POWER = 23.0     # dBm
    BASE_POWER = 50.0       # Watts (active base consumption)
    IDLE_POWER = 15.0       # Watts (sleep mode)
    
    # QoS thresholds
    DROP_CALL_THRESHOLD = 0.05   # 5%
    LATENCY_THRESHOLD = 50.0     # ms
    CPU_THRESHOLD = 0.90         # 90%
    PRB_THRESHOLD = 0.90         # 90%
    
    # Traffic parameters
    TRAFFIC_LAMBDA = 5.0      # Poisson parameter
    PEAK_HOUR_MULTIPLIER = 1.5
    
    # Channel parameters
    BANDWIDTH_MHZ = 20.0
    NOISE_FIGURE_DB = 7.0
    THERMAL_NOISE_DBM = -174.0  # dBm/Hz
    SHADOWING_STD_DB = 4.0      # Log-normal shadowing σ
    SINR_DROP_THRESHOLD = -6.0  # dB (SINR < -6dB → drop)
    MAX_LATENCY = 200.0         # ms
    
    # Room dimensions
    ROOM_WIDTH = 120.0   # meters
    ROOM_HEIGHT = 50.0   # meters
    ROOM_DEPTH = 3.0     # meters
    
    def __init__(self, seed: int = 42, use_mobility_file: bool = True):
        """Initialize simulator"""
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        
        # Load cell layout
        self.cells = self._load_cell_layout()
        
        # Load or generate mobility traces
        if use_mobility_file:
            self.mobility_traces = self._load_mobility_traces()
        else:
            self.mobility_traces = None
        
        # Initialize UEs
        self.ues = self._init_ues()
        
        # Shadowing matrix (cell × UE) - persistent
        self.shadowing = self.rng.normal(0, self.SHADOWING_STD_DB, 
                                         (self.TOTAL_CELLS, self.TOTAL_UES))
        
        # Tracking metrics
        self.total_energy = 0.0
        self.cpu_violations = 0
        self.prb_violations = 0
        
        # Previous actions for transition delay
        self.prev_power_ratios = np.ones(self.TOTAL_CELLS)
    
    def _load_cell_layout(self) -> List[CellState]:
        """Load cell positions from JSON file"""
        layout_path = os.path.join(os.path.dirname(__file__), "cells_layout.json")
        
        if os.path.exists(layout_path):
            with open(layout_path, 'r') as f:
                data = json.load(f)
            cells = []
            for cell_data in data['cells']:
                cells.append(CellState(
                    id=cell_data['id'],
                    x=cell_data['x'],
                    y=cell_data['y'],
                    z=cell_data['z']
                ))
            return cells
        else:
            # Fallback: generate 2x6 grid
            return self._generate_cell_layout()
    
    def _generate_cell_layout(self) -> List[CellState]:
        """Generate 2x6 grid layout"""
        cells = []
        rows, cols = 2, 6
        x_spacing = self.ROOM_WIDTH / (cols + 1)
        y_spacing = self.ROOM_HEIGHT / (rows + 1)
        
        cell_id = 0
        for row in range(rows):
            for col in range(cols):
                x = x_spacing * (col + 1)
                y = y_spacing * (row + 1)
                cells.append(CellState(id=cell_id, x=x, y=y, z=self.ROOM_DEPTH))
                cell_id += 1
        return cells
    
    def _load_mobility_traces(self) -> Optional[np.ndarray]:
        """Load pre-generated mobility traces"""
        trace_path = os.path.join(os.path.dirname(__file__), "ue_mobility_trace.npy")
        
        if os.path.exists(trace_path):
            traces = np.load(trace_path)
            print(f"Loaded mobility traces: {traces.shape}")
            return traces
        else:
            print("Warning: Mobility traces not found, using random walk")
            return None
    
    def _init_ues(self) -> List[UEState]:
        """Initialize UEs"""
        ues = []
        for i in range(self.TOTAL_UES):
            if self.mobility_traces is not None:
                x, y, z = self.mobility_traces[0, i]
            else:
                x = self.rng.uniform(0, self.ROOM_WIDTH)
                y = self.rng.uniform(0, self.ROOM_HEIGHT)
                z = 0.0
            
            ues.append(UEState(id=i, x=x, y=y, z=z))
        return ues
    
    def reset(self, seed: int = None) -> Dict:
        """Reset simulator to initial state"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.current_step = 0
        self.total_energy = 0.0
        self.cpu_violations = 0
        self.prb_violations = 0
        self.prev_power_ratios = np.ones(self.TOTAL_CELLS)
        
        # Reset cells
        for cell in self.cells:
            cell.txPower = self.MAX_TX_POWER
            cell.power_ratio = 1.0
            cell.is_sleeping = False
            cell.wake_up_delay = 0
            cell.numConnectedUEs = 0
            cell.cpuUsage = 0.0
            cell.prbUsage = 0.0
        
        # Reset UEs with initial positions
        for i, ue in enumerate(self.ues):
            if self.mobility_traces is not None:
                ue.x, ue.y, ue.z = self.mobility_traces[0, i]
            else:
                ue.x = self.rng.uniform(0, self.ROOM_WIDTH)
                ue.y = self.rng.uniform(0, self.ROOM_HEIGHT)
            ue.serving_cell = -1
            ue.is_dropped = False
        
        # Regenerate shadowing
        self.shadowing = self.rng.normal(0, self.SHADOWING_STD_DB,
                                         (self.TOTAL_CELLS, self.TOTAL_UES))
        
        # Initial assignments
        self._assign_ues_to_cells()
        self._calculate_metrics()
        
        return self.get_state()
    
    def step(self, power_ratios: np.ndarray) -> Dict:
        """
        Execute one simulation step
        
        Args:
            power_ratios: Array of 12 values in [0, 1]
                         Maps to Tx Power: PR * (max - min) + min
        """
        self.current_step += 1
        
        # 1. Apply power actions
        self._apply_power_actions(power_ratios)
        
        # 2. Move UEs
        self._move_ues()
        
        # 3. Assign UEs to cells
        self._assign_ues_to_cells()
        
        # 4. Calculate all metrics
        self._calculate_metrics()
        
        # Update previous actions
        self.prev_power_ratios = power_ratios.copy()
        
        return self.get_state()
    
    def _apply_power_actions(self, power_ratios: np.ndarray):
        """Apply power ratio actions with transition delay"""
        for i, (cell, ratio) in enumerate(zip(self.cells, power_ratios)):
            prev_ratio = self.prev_power_ratios[i]
            
            # Check for sleep → active transition
            was_sleeping = prev_ratio < 0.1
            wants_active = ratio >= 0.1
            
            if was_sleeping and wants_active:
                # Start wake-up (1 step delay)
                cell.wake_up_delay = 1
                cell.is_sleeping = True
                cell.txPower = -100.0  # No transmission during wake-up
                cell.power_ratio = ratio
            elif cell.wake_up_delay > 0:
                # Still waking up
                cell.wake_up_delay -= 1
                if cell.wake_up_delay == 0:
                    cell.is_sleeping = False
                    cell.txPower = self._ratio_to_power(ratio)
            elif ratio < 0.1:
                # Sleep mode
                cell.is_sleeping = True
                cell.txPower = -100.0  # No transmission
                cell.power_ratio = ratio
            else:
                # Active mode
                cell.is_sleeping = False
                cell.txPower = self._ratio_to_power(ratio)
                cell.power_ratio = ratio
    
    def _ratio_to_power(self, ratio: float) -> float:
        """Convert ratio [0.1-1.0] to Tx power [10-23 dBm]"""
        # From mentor: newTxPower = PowerRatio * (max - min) + min
        return ratio * (self.MAX_TX_POWER - self.MIN_TX_POWER) + self.MIN_TX_POWER
    
    def _move_ues(self):
        """Update UE positions from mobility traces or random walk"""
        t = min(self.current_step, self.SIM_TIME - 1)
        
        for i, ue in enumerate(self.ues):
            if self.mobility_traces is not None:
                ue.x, ue.y, ue.z = self.mobility_traces[t, i]
            else:
                # Random walk fallback
                angle = self.rng.uniform(0, 2 * np.pi)
                ue.x = np.clip(ue.x + np.cos(angle), 0, self.ROOM_WIDTH)
                ue.y = np.clip(ue.y + np.sin(angle), 0, self.ROOM_HEIGHT)
    
    def _calculate_path_loss(self, distance: float) -> float:
        """3GPP Indoor Office LOS path loss model - CORRECTED"""
        d = max(distance, 1.0)
        # Simplified indoor path loss: PL = 38.46 + 20*log10(d) for indoor
        # Using simpler model for realistic indoor coverage
        pl = 38.46 + 20 * np.log10(d)
        return pl
    
    def _calculate_rsrp(self, ue: UEState, cell: CellState) -> float:
        """Calculate RSRP including shadowing and antenna gain"""
        distance = np.sqrt((ue.x - cell.x)**2 + (ue.y - cell.y)**2 + (ue.z - cell.z)**2)
        path_loss = self._calculate_path_loss(distance)
        shadowing = self.shadowing[cell.id, ue.id]
        antenna_gain = 5.0  # dBi typical for small cell
        rsrp = cell.txPower + antenna_gain - path_loss - shadowing
        return rsrp
    
    def _assign_ues_to_cells(self):
        """Assign each UE to cell with max RSRP"""
        # Reset cell UE counts
        for cell in self.cells:
            cell.numConnectedUEs = 0
        
        for ue in self.ues:
            best_rsrp = -np.inf
            best_cell = -1
            
            for cell in self.cells:
                if cell.is_sleeping or cell.txPower < -50:
                    continue
                
                rsrp = self._calculate_rsrp(ue, cell)
                if rsrp > best_rsrp:
                    best_rsrp = rsrp
                    best_cell = cell.id
            
            ue.serving_cell = best_cell
            ue.rsrp = best_rsrp if best_cell >= 0 else -100.0
            ue.is_dropped = (best_cell < 0)
            
            if best_cell >= 0:
                self.cells[best_cell].numConnectedUEs += 1
    
    def _calculate_metrics(self):
        """Calculate all KPIs and metrics"""
        # Noise power
        noise_power_dbm = (self.THERMAL_NOISE_DBM + 
                          10 * np.log10(self.BANDWIDTH_MHZ * 1e6) + 
                          self.NOISE_FIGURE_DB)
        noise_linear = 10 ** (noise_power_dbm / 10)
        
        # Calculate per-UE SINR and latency
        for ue in self.ues:
            if ue.is_dropped or ue.serving_cell < 0:
                ue.sinr = -20.0
                ue.latency = self.MAX_LATENCY
                continue
            
            serving_cell = self.cells[ue.serving_cell]
            signal = 10 ** (ue.rsrp / 10)
            
            # Interference from other active cells
            interference = 0.0
            for cell in self.cells:
                if cell.id == ue.serving_cell or cell.is_sleeping:
                    continue
                if cell.txPower > -50:
                    int_rsrp = self._calculate_rsrp(ue, cell)
                    interference += 10 ** (int_rsrp / 10)
            
            sinr_linear = signal / (interference + noise_linear)
            ue.sinr = 10 * np.log10(max(sinr_linear, 1e-10))
            
            # Check for drop (SINR < -6 dB)
            if ue.sinr < self.SINR_DROP_THRESHOLD:
                ue.is_dropped = True
                ue.latency = self.MAX_LATENCY
            else:
                # Latency model: Base latency + queue delay based on cell load
                # Tuned for 5G indoor: target < 50ms under normal conditions
                serving_load = serving_cell.numConnectedUEs / 20.0  # 20 UEs = full load
                base_latency = 3.0  # 3ms base for 5G (reduced)
                queue_delay = 35.0 * min(serving_load, 1.0)  # Up to 35ms under load (reduced)
                # Power factor: lower power = higher latency
                power_ratio = (cell.txPower - self.MIN_TX_POWER) / (self.MAX_TX_POWER - self.MIN_TX_POWER)
                power_penalty = 15.0 * max(0, 0.7 - power_ratio)  # Penalty when power < 70%
                ue.latency = min(self.MAX_LATENCY, base_latency + queue_delay + power_penalty)
        
        # Calculate per-cell metrics
        for cell in self.cells:
            connected_ues = [ue for ue in self.ues if ue.serving_cell == cell.id]
            
            if connected_ues:
                cell.avgRSRP = np.mean([ue.rsrp for ue in connected_ues])
                cell.avgSINR = np.mean([ue.sinr for ue in connected_ues])
                cell.totalTrafficDemand = sum(ue.traffic_demand for ue in connected_ues)
            else:
                cell.avgRSRP = -100.0
                cell.avgSINR = 0.0
                cell.totalTrafficDemand = 0.0
            
            # Load and PRB usage
            cell.loadRatio = cell.numConnectedUEs / 20.0  # Assume 20 UEs per cell max
            cell.currentLoad = cell.loadRatio * cell.maxCapacity
            cell.prbUsage = min(1.0, cell.loadRatio)
            cell.cpuUsage = min(1.0, cell.loadRatio * 0.8)
            
            # Energy consumption
            if cell.is_sleeping:
                cell.energyConsumption = self.IDLE_POWER
            else:
                # Base power + dynamic based on tx power ratio
                power_factor = (cell.txPower - self.MIN_TX_POWER) / (self.MAX_TX_POWER - self.MIN_TX_POWER)
                cell.energyConsumption = self.BASE_POWER * (0.5 + 0.5 * power_factor)
            
            # Track violations
            if cell.cpuUsage > self.CPU_THRESHOLD:
                self.cpu_violations += 1
            if cell.prbUsage > self.PRB_THRESHOLD:
                self.prb_violations += 1
        
        # Update total energy
        step_energy = sum(c.energyConsumption for c in self.cells) / 1000.0  # kWh
        self.total_energy += step_energy * (self.TIME_STEP / 3600.0)
    
    def get_state(self) -> Dict:
        """Get complete state matching DataDescription format"""
        # Calculate network-level metrics
        dropped = sum(1 for ue in self.ues if ue.is_dropped)
        connected = self.TOTAL_UES - dropped
        
        latencies = [ue.latency for ue in self.ues if not ue.is_dropped]
        avg_latency = np.mean(latencies) if latencies else self.MAX_LATENCY
        
        active_cells = sum(1 for c in self.cells if not c.is_sleeping)
        
        return {
            # === Simulation Features (17) ===
            'totalCells': self.TOTAL_CELLS,
            'totalUEs': self.TOTAL_UES,
            'simTime': self.SIM_TIME,
            'timeStep': self.current_step,
            'timeProgress': self.current_step / self.SIM_TIME,
            'carrierFrequency': self.CARRIER_FREQUENCY,
            'isd': self.ISD,
            'minTxPower': self.MIN_TX_POWER,
            'maxTxPower': self.MAX_TX_POWER,
            'basePower': self.BASE_POWER,
            'idlePower': self.IDLE_POWER,
            'dropCallThreshold': self.DROP_CALL_THRESHOLD,
            'latencyThreshold': self.LATENCY_THRESHOLD,
            'cpuThreshold': self.CPU_THRESHOLD,
            'prbThreshold': self.PRB_THRESHOLD,
            'trafficLambda': self.TRAFFIC_LAMBDA,
            'peakHourMultiplier': self.PEAK_HOUR_MULTIPLIER,
            
            # === Network Features (12) ===
            'totalEnergy': self.total_energy,
            'activeCells': active_cells,
            'avgDropRate': dropped / self.TOTAL_UES,
            'avgLatency': avg_latency,
            'totalTraffic': sum(c.totalTrafficDemand for c in self.cells),
            'connectedUEs': connected,
            'cpuViolations': self.cpu_violations,
            'prbViolations': self.prb_violations,
            'maxCpuUsage': max(c.cpuUsage for c in self.cells),
            'maxPrbUsage': max(c.prbUsage for c in self.cells),
            'totalTxPower': sum(c.txPower for c in self.cells if not c.is_sleeping),
            'avgPowerRatio': np.mean([c.power_ratio for c in self.cells]),
            
            # === Cell Features (12 × 12 = 144) ===
            'cells': [
                {
                    'id': c.id,
                    'cpuUsage': c.cpuUsage,
                    'prbUsage': c.prbUsage,
                    'currentLoad': c.currentLoad,
                    'maxCapacity': c.maxCapacity,
                    'numConnectedUEs': c.numConnectedUEs,
                    'txPower': c.txPower,
                    'energyConsumption': c.energyConsumption,
                    'avgRSRP': c.avgRSRP,
                    'avgRSRQ': c.avgRSRQ,
                    'avgSINR': c.avgSINR,
                    'totalTrafficDemand': c.totalTrafficDemand,
                    'loadRatio': c.loadRatio,
                    'is_sleeping': c.is_sleeping,
                }
                for c in self.cells
            ],
            
            # === Additional computed metrics ===
            'qos_violation': (dropped / self.TOTAL_UES > self.DROP_CALL_THRESHOLD or 
                             avg_latency > self.LATENCY_THRESHOLD),
            'power_saved_percent': (1 - sum(c.energyConsumption for c in self.cells) / 
                                   (self.TOTAL_CELLS * self.BASE_POWER)) * 100,
        }
    
    def get_flat_observation(self) -> np.ndarray:
        """Get flattened observation vector (173 dimensions)"""
        state = self.get_state()
        obs = []
        
        # Simulation Features (17)
        obs.extend([
            state['totalCells'] / 20,
            state['totalUEs'] / 200,
            state['simTime'] / 500,
            state['timeStep'] / state['simTime'],
            state['timeProgress'],
            state['carrierFrequency'] / 10,
            state['isd'] / 50,
            state['minTxPower'] / 30,
            state['maxTxPower'] / 30,
            state['basePower'] / 100,
            state['idlePower'] / 50,
            state['dropCallThreshold'],
            state['latencyThreshold'] / 100,
            state['cpuThreshold'],
            state['prbThreshold'],
            state['trafficLambda'] / 10,
            state['peakHourMultiplier'] / 3,
        ])
        
        # Network Features (12)
        obs.extend([
            min(state['totalEnergy'], 10) / 10,
            state['activeCells'] / self.TOTAL_CELLS,
            state['avgDropRate'],
            state['avgLatency'] / self.MAX_LATENCY,
            min(state['totalTraffic'], 500) / 500,
            state['connectedUEs'] / self.TOTAL_UES,
            min(state['cpuViolations'], 100) / 100,
            min(state['prbViolations'], 100) / 100,
            state['maxCpuUsage'],
            state['maxPrbUsage'],
            max(state['totalTxPower'], 0) / (self.TOTAL_CELLS * self.MAX_TX_POWER),
            state['avgPowerRatio'],
        ])
        
        # Cell Features (12 × 12 = 144)
        for cell in state['cells']:
            obs.extend([
                cell['cpuUsage'],
                cell['prbUsage'],
                cell['currentLoad'] / cell['maxCapacity'],
                cell['maxCapacity'] / 500,
                cell['numConnectedUEs'] / 20,
                (cell['txPower'] + 100) / 123,  # Normalize including -100
                cell['energyConsumption'] / 100,
                (cell['avgRSRP'] + 120) / 80,
                (cell['avgRSRQ'] + 20) / 20,
                (cell['avgSINR'] + 20) / 50,
                cell['totalTrafficDemand'] / 50,
                cell['loadRatio'],
            ])
        
        return np.array(obs, dtype=np.float32)


if __name__ == "__main__":
    print("Testing Indoor Hotspot Simulator V2...")
    print("=" * 60)
    
    sim = IndoorHotspotSimulatorV2(seed=42)
    state = sim.reset()
    
    print(f"\nSimulation Config:")
    print(f"  Cells: {state['totalCells']}")
    print(f"  UEs: {state['totalUEs']}")
    print(f"  Sim Time: {state['simTime']}s")
    
    print(f"\nInitial State:")
    print(f"  Active cells: {state['activeCells']}/{state['totalCells']}")
    print(f"  Connected UEs: {state['connectedUEs']}/{state['totalUEs']}")
    print(f"  Avg Drop Rate: {state['avgDropRate']*100:.2f}%")
    print(f"  Avg Latency: {state['avgLatency']:.1f}ms")
    
    # Test flat observation
    obs = sim.get_flat_observation()
    print(f"\nFlat Observation: shape={obs.shape}, range=[{obs.min():.3f}, {obs.max():.3f}]")
    
    # Test with full power
    print(f"\n--- Test: Full Power ---")
    actions = np.ones(12)
    state = sim.step(actions)
    print(f"  Drop Rate: {state['avgDropRate']*100:.2f}%")
    print(f"  Avg Latency: {state['avgLatency']:.1f}ms")
    print(f"  Power Saved: {state['power_saved_percent']:.1f}%")
    
    # Test with half sleeping
    print(f"\n--- Test: Half Cells Sleep ---")
    actions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    state = sim.step(actions)
    print(f"  Active cells: {state['activeCells']}")
    print(f"  Drop Rate: {state['avgDropRate']*100:.2f}%")
    print(f"  QoS Violation: {state['qos_violation']}")
    
    print(f"\n✓ Simulator V2 test completed!")
