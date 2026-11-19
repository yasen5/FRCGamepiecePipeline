"""Geometry shared by the field renderer and layout generator."""
from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat

from .camera import Matrix, transform


class FieldRotation(BaseModel):
    axis: Literal['x', 'y', 'z']
    degrees: float


class FieldPiece(BaseModel):
    stagedObjects: list[str] = Field(default_factory=list)


class FieldConfig(BaseModel):
    widthInches: PositiveFloat
    heightInches: PositiveFloat
    rotations: list[FieldRotation]
    gamePieces: list[FieldPiece] = Field(default_factory=list)

    @property
    def size_m(self) -> tuple[float, float]:
        return self.widthInches * .0254, self.heightInches * .0254

    def world_transform(self) -> Matrix:
        width, height = self.size_m
        pose = transform((width / 2, height / 2, 0), (0, 0, 180))
        for rotation in self.rotations:
            angles = tuple(rotation.degrees if axis == rotation.axis else 0. for axis in 'xyz')
            pose = pose @ transform((0, 0, 0), (angles[0], angles[1], angles[2]))
        return pose

    def staged_nodes(self, nodes: list[str]) -> set[str]:
        staged = {name.replace(' ', '_') for piece in self.gamePieces for name in piece.stagedObjects}
        return {node for node in nodes if node.replace(' ', '_') in staged}
