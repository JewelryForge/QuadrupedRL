import pybullet as p


def analyse_joint_info(joint_info: tuple):
    analysis = []
    joint_types = {p.JOINT_REVOLUTE: 'revolute', p.JOINT_PRISMATIC: 'prismatic', p.JOINT_SPHERICAL: 'spherical',
                   p.JOINT_PLANAR: 'planar', p.JOINT_FIXED: 'fixed'}
    analysis.append(f'{joint_info[0]}_{joint_info[1].decode("UTF-8")}: {joint_types[joint_info[2]]}')
    if joint_info[3] != -1 and joint_info[4] != -1:
        analysis.append(f'q{joint_info[3]} u{joint_info[4]}')
        analysis.append(f'[{joint_info[8]:.2f}, {joint_info[9]:.2f}]')
    analysis.append(joint_info[12].decode("UTF-8"))
    analysis.append(f'axis {joint_info[13]}')
    analysis.append(f'pos {joint_info[14]}')
    analysis.append(f'orn {joint_info[15]}')
    analysis.append(f'parent {joint_info[16]}')

    return ', '.join(analysis)
