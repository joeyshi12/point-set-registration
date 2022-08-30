import numpy as np
from rigid_point_set_registration import rigid_point_set_registration
from scaling_point_set_registration import scaling_point_set_registration


def main():
    theta = np.linspace(0, 2 * np.pi)
    source = np.column_stack([4 * np.cos(theta), np.sin(theta)])
    angle = 1
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    t = np.random.randn(2)
    target = np.column_stack([np.cos(theta), np.sin(theta)]) @ R.T + t
    rigid_point_set_registration(source, target)


if __name__ == "__main__":
    main()
