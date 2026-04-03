import numpy as np


class DroneState:
    """
    Parses the raw 20-element state vector from BaseAviary._getDroneStateVector()
    into named fields.

    State vector layout (20 values):
        [0:3]   pos       - x, y, z position (meters)
        [3:7]   quat      - quaternion orientation (x, y, z, w)
        [7:10]  rpy       - roll, pitch, yaw (radians)
        [10:13] vel       - linear velocity x, y, z (m/s)
        [13:16] ang_v     - angular velocity x, y, z (rad/s)
        [16:20] rotor_rpms  - last clipped RPM for each rotor (0=front-left, 1=front-right, 2=rear-left, 3=rear-right)
    """

    def __init__(self, state_vector: np.ndarray):
        if state_vector.shape != (20,):
            raise ValueError(f"Expected state vector of shape (20,), got {state_vector.shape}")

        self.pos        = state_vector[0:3]    # x, y, z
        self.quat       = state_vector[3:7]    # quaternion (x, y, z, w)
        self.rpy        = state_vector[7:10]   # roll, pitch, yaw in radians
        self.vel        = state_vector[10:13]  # linear velocity
        self.ang_v      = state_vector[13:16]  # angular velocity
        self.rotor_rpms = state_vector[16:20]  # RPM per rotor: [front-left, front-right, rear-left, rear-right]

    @property
    def x(self) -> float:
        return float(self.pos[0])

    @property
    def y(self) -> float:
        return float(self.pos[1])

    @property
    def z(self) -> float:
        return float(self.pos[2])

    @property
    def roll(self) -> float:
        return float(self.rpy[0])

    @property
    def pitch(self) -> float:
        return float(self.rpy[1])

    @property
    def yaw(self) -> float:
        return float(self.rpy[2])

    @property
    def rotor0_rpm(self) -> float:
        """Front-left rotor RPM."""
        return float(self.rotor_rpms[0])

    @property
    def rotor1_rpm(self) -> float:
        """Front-right rotor RPM."""
        return float(self.rotor_rpms[1])

    @property
    def rotor2_rpm(self) -> float:
        """Rear-left rotor RPM."""
        return float(self.rotor_rpms[2])

    @property
    def rotor3_rpm(self) -> float:
        """Rear-right rotor RPM."""
        return float(self.rotor_rpms[3])

    @property
    def speed(self) -> float:
        """Scalar speed (magnitude of linear velocity)."""
        return float(np.linalg.norm(self.vel))

    def distance_to(self, target: np.ndarray) -> float:
        """Euclidean distance from current position to a target (x, y, z)."""
        return float(np.linalg.norm(self.pos - np.array(target)))

    def __repr__(self) -> str:
        return (
            f"DroneState(\n"
            f"  pos    = [{self.x:+.3f}, {self.y:+.3f}, {self.z:+.3f}] m\n"
            f"  rpy    = [{self.roll:+.3f}, {self.pitch:+.3f}, {self.yaw:+.3f}] rad\n"
            f"  vel    = [{self.vel[0]:+.3f}, {self.vel[1]:+.3f}, {self.vel[2]:+.3f}] m/s\n"
            f"  ang_v  = [{self.ang_v[0]:+.3f}, {self.ang_v[1]:+.3f}, {self.ang_v[2]:+.3f}] rad/s\n"
            f"  speed  = {self.speed:.3f} m/s\n"
            f"  rotors = [{self.rotor0_rpm:.1f}, {self.rotor1_rpm:.1f}, {self.rotor2_rpm:.1f}, {self.rotor3_rpm:.1f}] RPM\n"
            f")"
        )


def get_drone_state(env, drone_index: int = 0) -> DroneState:
    """
    Retrieve the current state of a drone from a running gym-pybullet-drones environment.

    Parameters
    ----------
    env : BaseAviary (or subclass)
        The active simulation environment.
    drone_index : int
        Index of the drone to query (default 0 for single-drone envs).

    Returns
    -------
    DroneState
        Parsed state object with named fields.
    """
    raw = env._getDroneStateVector(drone_index)
    return DroneState(raw)
