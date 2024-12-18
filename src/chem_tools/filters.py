"""Functions to filter molecules based on various criteria."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Concatenate, Literal, ParamSpec, overload

from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

from chem_tools._conversion import check_mol
from chem_tools._parallel import auto_parallel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

P = ParamSpec("P")

PAINS = Path(__file__).parent / "assets/pains.txt"
PAINS_FILE_COLUMNS = 2
UNWANTED_ELEMENTS = ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I")  # fmt: skip # noqa: E501
NEUTRALISATION_PATTERNS = (
    # Imidazoles
    ("[n+;H]", "n"),
    # Amines
    ("[N+;!H0]", "N"),
    # Carboxylic acids and alcohols
    ("[$([O-]);!$([O-][#7])]", "O"),
    # Thiols
    ("[S-;X1]", "S"),
    # Sulfonamides
    ("[$([N-;X2]S(=O)=O)]", "N"),
    # Enamines
    ("[$([N-;X2][C,N]=C)]", "N"),
    # Tetrazoles
    ("[n-]", "[nH]"),
    # Sulfoxides
    ("[$([S-]=O)]", "S"),
    # Amides
    ("[$([N-]C=O)]", "N"),
)


@functools.lru_cache(maxsize=1)
def _load_pains() -> dict[str, str]:
    """Load the PAINS SMARTS strings from the pains.txt file.

    Returns:
        A dictionary of PAINS SMARTS strings and their descriptions.

    Raises:
        ValueError: If not two space-seperated columns in pains.txt
    """
    pains = PAINS.read_text("utf-8").splitlines()

    sub_strct: list[list[str]] = [line.rstrip().split(" ") for line in pains if line]

    if any(len(x) != PAINS_FILE_COLUMNS for x in sub_strct):
        raise ValueError("Not two space-seperated columns in pains.txt")

    return {x[0]: x[1] for x in sub_strct}


def _filter_pains(mol: Chem.Mol, pains: Sequence[str]) -> bool:
    """Returns True if the molecule does not contain any PAINS substructure."""
    for smarts in pains:
        subs = Chem.MolFromSmarts(smarts)
        if subs.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMARTS: {smarts}")

        if mol.HasSubstructMatch(subs):
            return False
    return True


def filter_pains(mols: Sequence[Chem.Mol], n_jobs: int) -> list[Chem.Mol]:
    """Filter a sequence of molecules for PAINS.

    Args:
        mols: A sequence of molecules.
        n_jobs: The number of jobs to run in parallel.

    Returns:
        A list of molecules that do not contain PAINS.
    """
    pains = _load_pains()
    pains_smarts = list(pains.keys())

    filtered = auto_parallel(
        _filter_pains, ((mol, pains_smarts) for mol in mols), n_jobs=n_jobs
    )
    return [mol for mol, f in zip(mols, filtered, strict=True) if f]


def check_mol_for_change(
    mol: Chem.Mol,
    func: Callable[Concatenate[Chem.Mol, P], Chem.Mol | None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[Chem.Mol, bool] | None:
    """Check if a function changes a molecule.

    Args:
        mol: A molecule.
        func: A function that takes a molecule as its first argument
            and returns a modified molecule or None if the operation failed.
        *args: Additional arguments
        **kwargs: Additional keyword arguments

    Returns:
        A tuple containing the molecule and a boolean
        indicating whether the SMILES string has changed
        or None if the molecule is invalid.
    """
    smi = Chem.MolToSmiles(mol)
    modified_mol = func(mol, *args, **kwargs)

    if modified_mol is None:
        return None

    modified_smi = Chem.MolToSmiles(modified_mol)
    changed = smi != modified_smi
    return modified_mol, changed


def remove_salts(mol: Chem.Mol) -> Chem.Mol:
    """Remove salts from a molecule.

    Args:
        mol: A molecule.

    Returns:
        A molecule with salts removed.
    """
    remover = SaltRemover.SaltRemover()  # type: ignore[no-untyped-call]
    return remover.StripMol(mol)  # type: ignore[no-untyped-call, no-any-return]


def choose_largest_fragment(mol: Chem.Mol) -> Chem.Mol | None:
    """Choose the largest fragment from a molecule.

    Args:
        mol: A molecule.

    Returns:
        A molecule with the largest fragment or None if the molecule is invalid.

    Raises:
        ValueError: If the molecule is invalid.
    """
    chooser = rdMolStandardize.LargestFragmentChooser()
    mol = chooser.choose(mol)
    if "." in Chem.MolToSmiles(mol):
        raise ValueError("Molecule still contains multiple fragments")

    if check_mol(mol):
        return mol

    return None


@overload
def remove_isotopes(mol: Chem.Mol, return_info: Literal[False] = False) -> Chem.Mol: ...


@overload
def remove_isotopes(
    mol: Chem.Mol, return_info: Literal[True]
) -> tuple[Chem.Mol, bool]: ...


@overload
def remove_isotopes(
    mol: Chem.Mol, return_info: bool = False
) -> Chem.Mol | tuple[Chem.Mol, bool]: ...


def remove_isotopes(
    mol: Chem.Mol, return_info: bool = False
) -> Chem.Mol | tuple[Chem.Mol, bool]:
    """Remove isotopes from a molecule.

    Args:
        mol: A molecule.
        return_info: Whether to return a boolean indicating
            if the molecule changed. Defaults to False.

    Returns:
        The molecule with isotopes removed or a tuple with the molecule
        and a boolean indicating whether the molecule was changed.
    """
    changed = False
    atoms: list[Chem.Atom] = mol.GetAtoms()  # type: ignore[no-untyped-call,call-arg]
    isotopes: list[int] = [atom.GetIsotope() for atom in atoms]
    for atom, isotope in zip(atoms, isotopes, strict=True):
        if isotope:
            changed = True
            atom.SetIsotope(0)

    return (mol, changed) if return_info else mol


def build_filter_smarts(
    unwanted_elements: Sequence[str] = UNWANTED_ELEMENTS,
) -> Chem.Mol:
    """Build a filter smarts pattern for unwanted elements.

    Args:
        unwanted_elements: Atom symbols of unwanted elements.
            Defaults to H/B/C/N/O/F/Si/P/S/Cl/Se/Br/I.

    Returns:
        A smarts pattern for unwanted elements.
    """
    if not unwanted_elements:
        raise ValueError("No unwanted elements provided")

    pt = Chem.GetPeriodicTable()
    atomic_numbers = [pt.GetAtomicNumber(x) for x in unwanted_elements]

    # create smarts "[!#1!#5!#6!#7!#8!#9!#14!#15!#16!#17!#34!#35!#53]"
    smarts = f"[!#{'!#'.join(str(x) for x in atomic_numbers)}]"
    return Chem.MolFromSmarts(smarts)


def check_for_unwanted_elems(mol: Chem.Mol, filter_smarts: Chem.Mol) -> bool:
    """Check if a molecule contains unwanted elements.

    Args:
        mol: A molecule.
        filter_smarts: A smarts pattern for unwanted elements.

    Returns:
        Whether the molecule contains unwanted elements.
    """
    return mol.HasSubstructMatch(filter_smarts)


def _init_neutralisation_rxns() -> list[tuple[Chem.Mol, Chem.Mol]]:
    """Initialize the neutralization reactions. (chem_utils.NEUTRALISATION_PATTERNS)."""

    def _convert(pattern: str, repl: str) -> tuple[Chem.Mol, Chem.Mol]:
        """Convert a pattern and replacement to RDKit molecules."""
        pattern_mol = Chem.MolFromSmarts(pattern)
        mol = Chem.MolFromSmiles(repl, sanitize=False)
        return pattern_mol, mol

    return [_convert(x, y) for x, y in NEUTRALISATION_PATTERNS]


@overload
def neutralise_charges(
    mol: Chem.Mol,
    return_info: Literal[False] = False,
    reactions: Sequence[tuple[Chem.Mol, Chem.Mol]] | None = ...,
) -> Chem.Mol: ...


@overload
def neutralise_charges(
    mol: Chem.Mol,
    return_info: Literal[True],
    reactions: Sequence[tuple[Chem.Mol, Chem.Mol]] | None = ...,
) -> tuple[Chem.Mol, bool]: ...


@overload
def neutralise_charges(
    mol: Chem.Mol,
    return_info: bool = False,
    reactions: Sequence[tuple[Chem.Mol, Chem.Mol]] | None = ...,
) -> Chem.Mol | tuple[Chem.Mol, bool]: ...


def neutralise_charges(
    mol: Chem.Mol,
    return_info: bool = False,
    reactions: Sequence[tuple[Chem.Mol, Chem.Mol]] | None = None,
) -> Chem.Mol | tuple[Chem.Mol, bool]:
    """Neutralise charges on a molecule based on a list of reactions patterns.

    Args:
        mol: A molecule.
        return_info: Whether to return a boolean indicating if the molecule changed.
            Defaults to False.
        reactions: A list of reaction patterns.
            Defaults to hem_utils.NEUTRALISATION_PATTERNS


    Returns:
        The neutralized molecule or a tuple with the molecule and a boolean indicating
        whether the molecule was changed.
    """
    if reactions is None:
        reactions = _init_neutralisation_rxns()

    replaced = False

    # repl = functools.partial(, mol=mol)

    for reactant, product in reactions:
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms: tuple[Chem.Mol]
            rms = Chem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]

    if replaced:
        Chem.SanitizeMol(mol)

    return (mol, replaced) if return_info else mol


def remove_stereo(mol: Chem.Mol) -> Chem.Mol:
    """Remove stereochemistry from a molecule (in-place).

    Args:
        mol: A molecule.

    Returns:
        The molecule with stereochemistry removed.
    """
    Chem.RemoveStereochemistry(mol)
    return mol


__all__ = [
    "NEUTRALISATION_PATTERNS",
    "UNWANTED_ELEMENTS",
    "build_filter_smarts",
    "check_for_unwanted_elems",
    "check_mol_for_change",
    "choose_largest_fragment",
    "filter_pains",
    "neutralise_charges",
    "remove_isotopes",
    "remove_salts",
    "remove_stereo",
]
