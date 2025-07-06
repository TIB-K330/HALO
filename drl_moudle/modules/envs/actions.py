from collections import namedtuple

ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
ActionRot = namedtuple('ActionRot', ['v', 'r'])
ActionAW = namedtuple('ActionAW', ['a', 'w'])
ActionDiff = namedtuple('ActionDiff', ['al', 'ar'])
