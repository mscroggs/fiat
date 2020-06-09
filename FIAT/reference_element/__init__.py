from .reference_element import Cell                                 # noqa: F401
from .simplex import (Simplex, Point, DefaultLine, DefaultTriangle, DefaultTetrahedron,
                      UFCInterval, UFCTriangle, UFCTetrahedron, ufc_simplex,
                      make_affine_mapping)                          # noqa: F401
from .cube import (TensorProductCell, UFCQuadrilateral, UFCHexahedron,
                   compute_unflattening_map, flatten_entities,
                   flatten_reference_cube)                          # noqa: F401
from .lattice import make_lattice                                   # noqa: F401

# Backwards compatible name
ReferenceElement = Simplex


def ufc_cell(cell):
    """Handle incoming calls from FFC."""

    # celltype could be a string or a cell.
    if isinstance(cell, str):
        celltype = cell
    else:
        celltype = cell.cellname()

    if " * " in celltype:
        # Tensor product cell
        return TensorProductCell(*map(ufc_cell, celltype.split(" * ")))
    elif celltype == "quadrilateral":
        return UFCQuadrilateral()
    elif celltype == "hexahedron":
        return UFCHexahedron()
    elif celltype == "interval":
        return ufc_simplex(1)
    elif celltype == "triangle":
        return ufc_simplex(2)
    elif celltype == "tetrahedron":
        return ufc_simplex(3)
    else:
        raise RuntimeError("Don't know how to create UFC cell of type %s" % str(celltype))
