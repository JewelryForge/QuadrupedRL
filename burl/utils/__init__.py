import pybullet as p


def tuple_compact_string(_tuple):
    return '(' + ' '.join(f'{f:.1f}' for f in _tuple) + ')'


def analyse_joint_info(joint_info: tuple):
    analysis = []
    joint_types = {p.JOINT_REVOLUTE: 'revolute', p.JOINT_PRISMATIC: 'prismatic', p.JOINT_SPHERICAL: 'spherical',
                   p.JOINT_PLANAR: 'planar', p.JOINT_FIXED: 'fixed'}
    analysis.append(f'{joint_info[0]}_{joint_info[1].decode("UTF-8")}: {joint_types[joint_info[2]]}')
    is_fixed = joint_info[2] == p.JOINT_FIXED
    if not is_fixed:
        # analysis.append(f'q{joint_info[3]} u{joint_info[4]}')
        analysis.append(f'damping {joint_info[6]} friction {joint_info[7]}')
        analysis.append(f'[{joint_info[8]:.2f}, {joint_info[9]:.2f}]')
    analysis.append(joint_info[12].decode("UTF-8"))
    if not is_fixed:
        axis_dict = {(1., 0., 0.): 'X', (0., 1., 0.): 'Y', (0., 0., 1.): 'Z',
                     (-1., 0., 0.): '-X', (0., -1., 0.): '-Y', (0., 0., -1.): '-Z', }
        if joint_info[13] in axis_dict:
            analysis.append(f'axis {axis_dict[joint_info[13]]}')
        else:
            analysis.append(f'axis{tuple_compact_string(joint_info[13])}')
    analysis.append(f'pos{tuple_compact_string(joint_info[14])}')
    analysis.append(f'orn{tuple_compact_string(joint_info[15])}')
    analysis.append(f'parent {joint_info[16]}')

    return ', '.join(analysis)
