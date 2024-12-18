"""This module contains the type hints for the `_tmap` module.

Generated with mypy's stubgen.
"""

# ruff: noqa: ANN001, ANN002, ANN003, D205, D402, D417, D418, E741, N802, N803, E501
# mypy: ignore-errors

from collections.abc import Iterable, Iterator
from typing import ClassVar, overload

import numpy as np

Absolute: ScalingType
Barycenter: Placer
Circle: Placer
EdgeCover: Merger
IndependentSet: Merger
LocalBiconnected: Merger
Median: Placer
Random: Placer
RelativeToAvgLength: ScalingType
RelativeToDesiredLength: ScalingType
RelativeToDrawing: ScalingType
Solar: Merger
Zero: Placer

class GraphProperties:
    def __init__(self) -> None:
        """__init__(self: _tmap.GraphProperties) -> None.

        Constructor for the class :obj:`GraphProperties`.

        """

    @property
    def adjacency_list(self) -> list[list[tuple[int, float]]]: ...
    @property
    def adjacency_list_knn(self) -> list[list[tuple[int, float]]]: ...
    @property
    def degrees(self) -> VectorUint: ...
    @property
    def mst_weight(self) -> float: ...
    @property
    def n_connected_components(self) -> int: ...
    @property
    def n_isolated_vertices(self) -> int: ...

class LSHForest:
    def __init__(
        self,
        d: int = ...,
        l: int = ...,
        store: bool = ...,
        file_backed: bool = ...,
        weighted: bool = ...,
    ) -> None:
        """__init__(self: _tmap.LSHForest, d: int = 128, l: int = 8, store: bool = True, file_backed: bool = False, weighted: bool = False) -> None.

        Constructor for the class :obj:`LSHForest`.

        Keyword Arguments:
            d (:obj:`int`): The dimensionality of the MinHashe vectors to be added
            l (:obj:`int`): The number of prefix trees used when indexing data
            store (:obj:`bool`) Whether to store the added MinHash vectors. This is required when using linear scan in queries
            file_backed (:obj:`bool`) Whether to store the data on disk rather than in main memory (experimental)

        """

    def add(self, arg0: VectorUint) -> None:
        """add(self: _tmap.LSHForest, arg0: _tmap.VectorUint) -> None.

        Add a MinHash vector to the LSH forest.

        Arguments:
            vecs (:obj:`VectorUint`): A MinHash vector that is to be added to the LSH forest

        """

    def batch_add(self, arg0: list[VectorUint]) -> None:
        """batch_add(self: _tmap.LSHForest, arg0: list[_tmap.VectorUint]) -> None.

        Add a list MinHash vectors to the LSH forest (parallelized).

        Arguments:
            vecs (:obj:`list` of :obj:`VectorUint`): A list of MinHash vectors that is to be added to the LSH forest

        """

    def batch_query(self, arg0: list[VectorUint], arg1: int) -> list[VectorUint]:
        """batch_query(self: _tmap.LSHForest, arg0: list[_tmap.VectorUint], arg1: int) -> list[_tmap.VectorUint].

        Query the LSH forest for k-nearest neighbors (parallelized).

        Arguments:
            vecs (:obj:`list` of :obj:`VectorUint`): The query MinHash vectors
            k (:obj:`int`): The number of nearest neighbors to be retrieved

        Returns:
            :obj:`list` of :obj:`VectorUint`: The results of the queries

        """

    def clear(self) -> None:
        """clear(self: _tmap.LSHForest) -> None.

        Clears all the added data and computed indices from this :obj:`LSHForest` instance.

        """

    def fit(self, arg0: list[VectorUint], arg1: VectorUint) -> None:
        """fit(self: _tmap.LSHForest, arg0: list[_tmap.VectorUint], arg1: _tmap.VectorUint) -> None.

        Add Minhashes with labels to this LSHForest (parallelized).

        Arguments:
            vecs (:obj:`list` of :obj:`VectorUint`): A list of MinHash vectors that is to be added to the LSH forest
            labels (:obj:`VectorUint`) A vector containing labels.

        """

    def get_all_distances(self, arg0: VectorUint) -> VectorFloat:
        """get_all_distances(self: _tmap.LSHForest, arg0: _tmap.VectorUint) -> _tmap.VectorFloat.

        Calculate the Jaccard distances of a MinHash vector to all indexed MinHash vectors.

        Arguments:
            vec (:obj:`VectorUint`): The query MinHash vector

        Returns:
            :obj:`list` of :obj:`float`: The Jaccard distances

        """

    def get_all_nearest_neighbors(self, k: int, kc: int = ...) -> VectorUint:
        """get_all_nearest_neighbors(self: _tmap.LSHForest, k: int, kc: int = 10) -> _tmap.VectorUint.

        Get the k-nearest neighbors of all indexed MinHash vectors.

        Arguments:
            k (:obj:`int`): The number of nearest neighbors to be retrieved

        Keyword Arguments:
            kc (:obj:`int`): The factor by which :obj:`k` is multiplied for LSH forest retreival

        Returns:
            :obj:`VectorUint` The ids of all k-nearest neighbors

        """

    def get_distance(self, arg0: VectorUint, arg1: VectorUint) -> float:
        """get_distance(self: _tmap.LSHForest, arg0: _tmap.VectorUint, arg1: _tmap.VectorUint) -> float.

        Calculate the Jaccard distance between two MinHash vectors.

        Arguments:
            vec_a (:obj:`VectorUint`): A MinHash vector
            vec_b (:obj:`VectorUint`): A MinHash vector

        Returns:
            :obj:`float` The Jaccard distance

        """

    def get_distance_by_id(self, arg0: int, arg1: int) -> float:
        """get_distance_by_id(self: _tmap.LSHForest, arg0: int, arg1: int) -> float.

        Calculate the Jaccard distance between two indexed MinHash vectors.

        Arguments:
            a (:obj:`int`): The id of an indexed MinHash vector
            b (:obj:`int`): The id of an indexed MinHash vector

        Returns:
            :obj:`float` The Jaccard distance

        """

    def get_hash(self, arg0: int) -> VectorUint:
        """get_hash(self: _tmap.LSHForest, arg0: int) -> _tmap.VectorUint.

        Retrieve the MinHash vector of an indexed entry given its index. The index is defined by order of insertion.

        Arguments:
            a (:obj:`int`): The id of an indexed MinHash vector

        Returns:
            :obj:`VectorUint` The MinHash vector

        """

    def get_knn_graph(
        self,
        _from: VectorUint,
        to: VectorUint,
        weight: VectorFloat,
        k: int,
        kc: int = ...,
    ) -> None:
        """get_knn_graph(self: _tmap.LSHForest, from: _tmap.VectorUint, to: _tmap.VectorUint, weight: _tmap.VectorFloat, k: int, kc: int = 10) -> None.

        Construct the k-nearest neighbor graph of the indexed MinHash vectors. It will be written to out parameters :obj:`from`, :obj:`to`, and :obj:`weight` as an edge list.

        Arguments:
            from (:obj:`VectorUint`): A vector to which the ids for the from vertices are written
            to (:obj:`VectorUint`): A vector to which the ids for the to vertices are written
            weight (:obj:`VectorFloat`): A vector to which the edge weights are written
            k (:obj:`int`): The number of nearest neighbors to be retrieved during the construction of the k-nearest neighbor graph

        Keyword Arguments:
            kc (:obj:`int`): The factor by which :obj:`k` is multiplied for LSH forest retreival

        """

    def get_weighted_distance(self, arg0: VectorUint, arg1: VectorUint) -> float:
        """get_weighted_distance(self: _tmap.LSHForest, arg0: _tmap.VectorUint, arg1: _tmap.VectorUint) -> float.

        Calculate the weighted Jaccard distance between two MinHash vectors.

        Arguments:
            vec_a (:obj:`VectorUint`): A weighted MinHash vector
            vec_b (:obj:`VectorUint`): A weighted MinHash vector

        Returns:
            :obj:`float` The Jaccard distance

        """

    def get_weighted_distance_by_id(self, arg0: int, arg1: int) -> float:
        """get_weighted_distance_by_id(self: _tmap.LSHForest, arg0: int, arg1: int) -> float.

        Calculate the Jaccard distance between two indexed weighted MinHash vectors.

        Arguments:
            a (:obj:`int`): The id of an indexed weighted MinHash vector
            b (:obj:`int`): The id of an indexed weighted MinHash vector

        Returns:
            :obj:`float` The weighted Jaccard distance

        """

    def index(self) -> None:
        """index(self: _tmap.LSHForest) -> None.

        Index the LSH forest. This has to be run after each time new MinHashes were added.

        """

    def is_clean(self) -> bool:
        """is_clean(self: _tmap.LSHForest) -> bool.

        Returns a boolean indicating whether or not the LSH forest has been indexed after the last MinHash vector was added.

        Returns:
            :obj:`bool`: :obj:`True` if :obj:`index()` has been run since MinHash vectors have last been added using :obj:`add()` or :obj:`batch_add()`. :obj:`False` otherwise

        """

    def linear_scan(
        self, vec: VectorUint, indices: VectorUint, k: int = ...
    ) -> list[tuple[float, int]]:
        """linear_scan(self: _tmap.LSHForest, vec: _tmap.VectorUint, indices: _tmap.VectorUint, k: int = 10) -> list[tuple[float, int]].

        Query a subset of indexed MinHash vectors using linear scan.

        Arguments:
            vec (:obj:`VectorUint`): The query MinHash vector
            indices (:obj:`VectorUint`) The ids of indexed MinHash vectors that define the subset to be queried

        Keyword Arguments:
            k (:obj:`int`): The number of nearest neighbors to be retrieved

        Returns:
            :obj:`list` of :obj:`tuple[float, int]`: The results of the query

        """

    def predict(
        self, vecs: list[VectorUint], k: int = ..., kc: int = ..., weighted: bool = ...
    ) -> VectorUint:
        """predict(self: _tmap.LSHForest, vecs: list[_tmap.VectorUint], k: int = 10, kc: int = 10, weighted: bool = False) -> _tmap.VectorUint.

        Predict labels of Minhashes using the kNN algorithm (parallelized).

        Arguments:
            vecs (:obj:`list` of :obj:`VectorUint`): A list of MinHash vectors that is to be added to the LSH forest
            k (:obj:`int`) The degree of the kNN algorithm
            kc (:obj:`int`) The scalar by which k is multiplied before querying the LSH
            weighted (:obj:`bool` Whether distances are used as weights by the knn algorithm)

        Returns:
            :obj:`VectorUint` The predicted labels

        """

    def query(self, arg0: VectorUint, arg1: int) -> VectorUint:
        """query(self: _tmap.LSHForest, arg0: _tmap.VectorUint, arg1: int) -> _tmap.VectorUint.

        Query the LSH forest for k-nearest neighbors.

        Arguments:
            vec (:obj:`VectorUint`): The query MinHash vector
            k (:obj:`int`): The number of nearest neighbors to be retrieved

        Returns:
            :obj:`VectorUint`: The results of the query

        """

    def query_by_id(self, arg0: int, arg1: int) -> VectorUint:
        """query_by_id(self: _tmap.LSHForest, arg0: int, arg1: int) -> _tmap.VectorUint.

        Query the LSH forest for k-nearest neighbors.

        Arguments:
            id (:obj:`int`): The id of an indexed MinHash vector
            k (:obj:`int`): The number of nearest neighbors to be retrieved

        Returns:
            :obj:`VectorUint`: The results of the query

        """

    def query_exclude(
        self, arg0: VectorUint, arg1: VectorUint, arg2: int
    ) -> VectorUint:
        """query_exclude(self: _tmap.LSHForest, arg0: _tmap.VectorUint, arg1: _tmap.VectorUint, arg2: int) -> _tmap.VectorUint.

        Query the LSH forest for k-nearest neighbors.

        Arguments:
            vec (:obj:`VectorUint`): The query MinHash vector
            exclude (:obj:`list` of :obj:`VectorUint`) A list of ids of indexed MinHash vectors to be excluded from the search
            k (:obj:`int`): The number of nearest neighbors to be retrieved

        Returns:
            :obj:`VectorUint`: The results of the query

        """

    def query_exclude_by_id(self, arg0: int, arg1: VectorUint, arg2: int) -> VectorUint:
        """query_exclude_by_id(self: _tmap.LSHForest, arg0: int, arg1: _tmap.VectorUint, arg2: int) -> _tmap.VectorUint.

        Query the LSH forest for k-nearest neighbors.

        Arguments:
            id (:obj:`int`): The id of an indexed MinHash vector
            exclude (:obj:`list` of :obj:`VectorUint`) A list of ids of indexed MinHash vectors to be excluded from the search
            k (:obj:`int`): The number of nearest neighbors to be retrieved

        Returns:
            :obj:`VectorUint`: The results of the query

        """

    def query_linear_scan(
        self, vec: VectorUint, k: int, kc: int = ...
    ) -> list[tuple[float, int]]:
        """query_linear_scan(self: _tmap.LSHForest, vec: _tmap.VectorUint, k: int, kc: int = 10) -> list[tuple[float, int]].

        Query k-nearest neighbors with a LSH forest / linear scan combination. :obj:`k`*:obj:`kc` nearest neighbors are searched for using LSH forest; from these, the :obj:`k` nearest neighbors are retrieved using linear scan.

        Arguments:
            vec (:obj:`VectorUint`): The query MinHash vector
            k (:obj:`int`): The number of nearest neighbors to be retrieved

        Keyword Arguments:
            kc (:obj:`int`): The factor by which :obj:`k` is multiplied for LSH forest retreival

        Returns:
            :obj:`list` of :obj:`tuple[float, int]`: The results of the query

        """

    def query_linear_scan_by_id(
        self, id: int, k: int, kc: int = ...
    ) -> list[tuple[float, int]]:
        """query_linear_scan_by_id(self: _tmap.LSHForest, id: int, k: int, kc: int = 10) -> list[tuple[float, int]].

        Query k-nearest neighbors with a LSH forest / linear scan combination. :obj:`k`*:obj:`kc` nearest neighbors are searched for using LSH forest; from these, the :obj:`k` nearest neighbors are retrieved using linear scan.

        Arguments:
            id (:obj:`int`): The id of an indexed MinHash vector
            k (:obj:`int`): The number of nearest neighbors to be retrieved

        Keyword Arguments:
            kc (:obj:`int`): The factor by which :obj:`k` is multiplied for LSH forest retreival

        Returns:
            :obj:`list` of :obj:`tuple[float, int]`: The results of the query

        """

    def query_linear_scan_exclude(
        self, vec: VectorUint, k: int, exclude: VectorUint, kc: int = ...
    ) -> list[tuple[float, int]]:
        """query_linear_scan_exclude(self: _tmap.LSHForest, vec: _tmap.VectorUint, k: int, exclude: _tmap.VectorUint, kc: int = 10) -> list[tuple[float, int]].

        Query k-nearest neighbors with a LSH forest / linear scan combination. :obj:`k`*:obj:`kc` nearest neighbors are searched for using LSH forest; from these, the :obj:`k` nearest neighbors are retrieved using linear scan.

        Arguments:
            vec (:obj:`VectorUint`): The query MinHash vector
            k (:obj:`int`): The number of nearest neighbors to be retrieved

        Keyword Arguments:
            exclude (:obj:`list` of :obj:`VectorUint`) A list of ids of indexed MinHash vectors to be excluded from the search
            kc (:obj:`int`): The factor by which :obj:`k` is multiplied for LSH forest retreival

        Returns:
            :obj:`list` of :obj:`tuple[float, int]`: The results of the query

        """

    def query_linear_scan_exclude_by_id(
        self, id: int, k: int, exclude: VectorUint, kc: int = ...
    ) -> list[tuple[float, int]]:
        """query_linear_scan_exclude_by_id(self: _tmap.LSHForest, id: int, k: int, exclude: _tmap.VectorUint, kc: int = 10) -> list[tuple[float, int]].

        Query k-nearest neighbors with a LSH forest / linear scan combination. :obj:`k`*:obj:`kc` nearest neighbors are searched for using LSH forest; from these, the :obj:`k` nearest neighbors are retrieved using linear scan.

        Arguments:
            id (:obj:`int`): The id of an indexed MinHash vector
            k (:obj:`int`): The number of nearest neighbors to be retrieved

        Keyword Arguments:
            exclude (:obj:`list` of :obj:`VectorUint`) A list of ids of indexed MinHash vectors to be excluded from the search
            kc (:obj:`int`): The factor by which :obj:`k` is multiplied for LSH forest retreival

        Returns:
            :obj:`list` of :obj:`tuple[float, int]`: The results of the query

        """

    def restore(self, arg0: str) -> None:
        """restore(self: _tmap.LSHForest, arg0: str) -> None.

        Deserializes a previously serialized (using :obj:`store()`) state into this instance of :obj:`LSHForest` and recreates the index.

        Arguments:
            path (:obj:`str`): The path to the file which is deserialized

        """

    def size(self) -> int:
        """size(self: _tmap.LSHForest) -> int.

        Returns the number of MinHash vectors in this LSHForest instance.

        Returns:
            :obj:`int`: The number of MinHash vectors

        """

    def store(self, arg0: str) -> None:
        """store(self: _tmap.LSHForest, arg0: str) -> None.

        Serializes the current state of this instance of :obj:`LSHForest` to the disk in binary format. The index is not serialized and has to be rebuilt after deserialization.

        Arguments:
            path (:obj:`str`): The path to which to searialize the file

        """

class LayoutConfiguration:
    fme_iterations: int
    fme_precision: int
    fme_randomize: bool
    fme_threads: int
    k: int
    kc: int
    merger: Merger
    merger_adjustment: int
    merger_factor: float
    mmm_repeats: int
    node_size: float
    placer: Placer
    sl_extra_scaling_steps: int
    sl_repeats: int
    sl_scaling_max: float
    sl_scaling_min: float
    sl_scaling_type: ScalingType
    def __init__(self) -> None:
        """__init__(self: _tmap.LayoutConfiguration) -> None.

        Constructor for the class :obj:`LayoutConfiguration`.

        """

class Merger:
    __members__: ClassVar[dict] = ...  # read-only
    EdgeCover: ClassVar[Merger] = ...
    IndependentSet: ClassVar[Merger] = ...
    LocalBiconnected: ClassVar[Merger] = ...
    Solar: ClassVar[Merger] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: _tmap.Merger, value: int) -> None."""

    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool."""

    def __ge__(self, other: object) -> bool:
        """__ge__(self: object, other: object) -> bool."""

    def __gt__(self, other: object) -> bool:
        """__gt__(self: object, other: object) -> bool."""

    def __hash__(self) -> int:
        """__hash__(self: object) -> int."""

    def __index__(self) -> int:
        """__index__(self: _tmap.Merger) -> int."""

    def __int__(self) -> int:
        """__int__(self: _tmap.Merger) -> int."""

    def __le__(self, other: object) -> bool:
        """__le__(self: object, other: object) -> bool."""

    def __lt__(self, other: object) -> bool:
        """__lt__(self: object, other: object) -> bool."""

    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool."""

    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Minhash:
    def __init__(self, d: int = ..., seed: int = ..., sample_size: int = ...) -> None:
        """__init__(self: _tmap.Minhash, d: int = 128, seed: int = 42, sample_size: int = 128) -> None.

        Constructor for the class :obj:`Minhash`.

        Keyword Arguments:
            d (:obj:`int`): The number of permutations used for hashing
            seed (:obj:`int`): The seed used for the random number generator(s)
            sample_size (:obj:`int`): The sample size when generating a weighted MinHash

        """

    @overload
    def batch_from_binary_array(self, arg0: list) -> list[VectorUint]:
        """batch_from_binary_array(*args, **kwargs)
        Overloaded function.

        1. batch_from_binary_array(self: _tmap.Minhash, arg0: list) -> list[_tmap.VectorUint]


                    Create MinHash vectors from binary arrays (parallelized).

        Arguments:
                        vec (:obj:`list` of :obj:`list`): A list of lists containing binary values

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        2. batch_from_binary_array(self: _tmap.Minhash, arg0: numpy.ndarray[numpy.uint8]) -> list[_tmap.VectorUint]


                 py::overload_cast<py::array_t<uint8_t>&>(&PyMinhash::BatchFromBinaryArray), R"pbdoc(
                    Create MinHash vectors from binary arrays (parallelized).

        Arguments:
                        vec (:obj:`Array`): A 2D array containing binary values

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        3. batch_from_binary_array(self: _tmap.Minhash, arg0: list[_tmap.VectorUchar]) -> list[_tmap.VectorUint]


                    Create MinHash vectors from binary arrays (parallelized).

        Arguments:
                        vec (:obj:`list` of :obj:`VectorUchar`): A list of vectors containing binary values

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors

        """

    @overload
    def batch_from_binary_array(self, arg0: np.ndarray[np.uint8]) -> list[VectorUint]:
        """batch_from_binary_array(*args, **kwargs)
        Overloaded function.

        1. batch_from_binary_array(self: _tmap.Minhash, arg0: list) -> list[_tmap.VectorUint]


                    Create MinHash vectors from binary arrays (parallelized).

        Arguments:
                        vec (:obj:`list` of :obj:`list`): A list of lists containing binary values

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        2. batch_from_binary_array(self: _tmap.Minhash, arg0: numpy.ndarray[numpy.uint8]) -> list[_tmap.VectorUint]


                 py::overload_cast<py::array_t<uint8_t>&>(&PyMinhash::BatchFromBinaryArray), R"pbdoc(
                    Create MinHash vectors from binary arrays (parallelized).

        Arguments:
                        vec (:obj:`Array`): A 2D array containing binary values

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        3. batch_from_binary_array(self: _tmap.Minhash, arg0: list[_tmap.VectorUchar]) -> list[_tmap.VectorUint]


                    Create MinHash vectors from binary arrays (parallelized).

        Arguments:
                        vec (:obj:`list` of :obj:`VectorUchar`): A list of vectors containing binary values

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors

        """

    def batch_from_int_weight_array(
        self, vecs: list[VectorUint], divide_by: int = ...
    ) -> list[VectorUint]:
        """batch_from_int_weight_array(self: _tmap.Minhash, vecs: list[_tmap.VectorUint], divide_by: int = 0) -> list[_tmap.VectorUint].

        Create MinHash vectors from :obj:`int` arrays, where entries are weights rather than indices of ones (parallelized).

        Arguments:
            vecs (:obj:`list` of :obj:`VectorUint`): A list of vectors containing :obj:`int` values
            divide_by (:obj:`int`): A integer by which each value of each vector is divided. Information is lost, but the running time will be lowered. Faster if :obj:`divide_by` is a power of two.

        Returns:
            :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors

        """

    def batch_from_sparse_binary_array(self, arg0: list) -> list[VectorUint]:
        """batch_from_sparse_binary_array(*args, **kwargs)
        Overloaded function.

        1. batch_from_sparse_binary_array(self: _tmap.Minhash, arg0: list) -> list[_tmap.VectorUint]


                 R"pbdoc(
                    Create MinHash vectors from sparse binary arrays (parallelized).

        Arguments:
                        vec (:obj:`list` of :obj:`list`): A list of Python lists containing indices of ones in a binary array

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        2. batch_from_sparse_binary_array(self: _tmap.Minhash, arg0: numpy.ndarray[numpy.uint32]) -> list[_tmap.VectorUint]


                 py::overload_cast<>(&PyMinhash::BatchFromSparseBinaryArray),
                 R"pbdoc(
                    Create MinHash vectors from sparse binary arrays (parallelized).

        Arguments:
                        vec (:obj:`Array`): A 2D array containing indices of ones in a binary array

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        3. batch_from_sparse_binary_array(self: _tmap.Minhash, arg0: list[_tmap.VectorUint]) -> list[_tmap.VectorUint]


                 R"pbdoc(
                    Create MinHash vectors from sparse binary arrays (parallelized).

        Arguments:
                        vec (:obj:`list` of :obj:`VectorUint`): A list of vectors containing indices of ones in a binary array

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors

        """

    def batch_from_string_array(self, arg0: list[list[str]]) -> list[VectorUint]:
        """batch_from_string_array(self: _tmap.Minhash, arg0: list[list[str]]) -> list[_tmap.VectorUint].

        Create MinHash vectors from string arrays (parallelized).

        Arguments:
            vec (:obj:`list` of :obj:`list` of :obj:`str`): A list of list of strings

        Returns:
            :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors

        """

    @overload
    def batch_from_weight_array(
        self, vecs: list, method: str = ...
    ) -> list[VectorUint]:
        """batch_from_weight_array(*args, **kwargs)
        Overloaded function.

        1. batch_from_weight_array(self: _tmap.Minhash, vecs: list, method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`list` of :obj:`list`): A list of Python lists containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        2. batch_from_weight_array(self: _tmap.Minhash, vecs: numpy.ndarray[numpy.float32], method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`Array`): A 2D array containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        3. batch_from_weight_array(self: _tmap.Minhash, vecs: numpy.ndarray[numpy.float64], method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`Array`): A 2D array containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        4. batch_from_weight_array(self: _tmap.Minhash, vecs: list[_tmap.VectorFloat], method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`list` of :obj:`VectorFloat`): A list of vectors containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors

        """

    @overload
    def batch_from_weight_array(
        self, vecs: np.ndarray[np.float32], method: str = ...
    ) -> list[VectorUint]:
        """batch_from_weight_array(*args, **kwargs)
        Overloaded function.

        1. batch_from_weight_array(self: _tmap.Minhash, vecs: list, method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`list` of :obj:`list`): A list of Python lists containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        2. batch_from_weight_array(self: _tmap.Minhash, vecs: numpy.ndarray[numpy.float32], method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`Array`): A 2D array containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        3. batch_from_weight_array(self: _tmap.Minhash, vecs: numpy.ndarray[numpy.float64], method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`Array`): A 2D array containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        4. batch_from_weight_array(self: _tmap.Minhash, vecs: list[_tmap.VectorFloat], method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`list` of :obj:`VectorFloat`): A list of vectors containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors

        """

    @overload
    def batch_from_weight_array(
        self, vecs: np.ndarray[np.float64], method: str = ...
    ) -> list[VectorUint]:
        """batch_from_weight_array(*args, **kwargs)
        Overloaded function.

        1. batch_from_weight_array(self: _tmap.Minhash, vecs: list, method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`list` of :obj:`list`): A list of Python lists containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        2. batch_from_weight_array(self: _tmap.Minhash, vecs: numpy.ndarray[numpy.float32], method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`Array`): A 2D array containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        3. batch_from_weight_array(self: _tmap.Minhash, vecs: numpy.ndarray[numpy.float64], method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`Array`): A 2D array containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        4. batch_from_weight_array(self: _tmap.Minhash, vecs: list[_tmap.VectorFloat], method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`list` of :obj:`VectorFloat`): A list of vectors containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors

        """

    @overload
    def batch_from_weight_array(
        self, vecs: list[VectorFloat], method: str = ...
    ) -> list[VectorUint]:
        """batch_from_weight_array(*args, **kwargs)
        Overloaded function.

        1. batch_from_weight_array(self: _tmap.Minhash, vecs: list, method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`list` of :obj:`list`): A list of Python lists containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        2. batch_from_weight_array(self: _tmap.Minhash, vecs: numpy.ndarray[numpy.float32], method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`Array`): A 2D array containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        3. batch_from_weight_array(self: _tmap.Minhash, vecs: numpy.ndarray[numpy.float64], method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`Array`): A 2D array containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors


        4. batch_from_weight_array(self: _tmap.Minhash, vecs: list[_tmap.VectorFloat], method: str = 'ICWS') -> list[_tmap.VectorUint]


                    Create MinHash vectors from :obj:`float` arrays (parallelized).

        Arguments:
                        vecs (:obj:`list` of :obj:`VectorFloat`): A list of vectors containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`list` of :obj:`VectorUint`: A list of MinHash vectors

        """

    @overload
    def from_binary_array(self, arg0: list) -> VectorUint:
        """from_binary_array(*args, **kwargs)
        Overloaded function.

        1. from_binary_array(self: _tmap.Minhash, arg0: list) -> _tmap.VectorUint


                    Create a MinHash vector from a binary array.

        Arguments:
                        vec (:obj:`list`): A Python list containing binary values

        Returns:
                        :obj:`VectorUint`: A MinHash vector


        2. from_binary_array(self: _tmap.Minhash, arg0: _tmap.VectorUchar) -> _tmap.VectorUint


                    Create a MinHash vector from a binary array.

        Arguments:
                        vec (:obj:`list`): A Python list containing binary values

        Returns:
                        :obj:`VectorUint`: A MinHash vector

        """

    @overload
    def from_binary_array(self, arg0: VectorUchar) -> VectorUint:
        """from_binary_array(*args, **kwargs)
        Overloaded function.

        1. from_binary_array(self: _tmap.Minhash, arg0: list) -> _tmap.VectorUint


                    Create a MinHash vector from a binary array.

        Arguments:
                        vec (:obj:`list`): A Python list containing binary values

        Returns:
                        :obj:`VectorUint`: A MinHash vector


        2. from_binary_array(self: _tmap.Minhash, arg0: _tmap.VectorUchar) -> _tmap.VectorUint


                    Create a MinHash vector from a binary array.

        Arguments:
                        vec (:obj:`list`): A Python list containing binary values

        Returns:
                        :obj:`VectorUint`: A MinHash vector

        """

    @overload
    def from_sparse_binary_array(self, arg0: list) -> VectorUint:
        """from_sparse_binary_array(*args, **kwargs)
        Overloaded function.

        1. from_sparse_binary_array(self: _tmap.Minhash, arg0: list) -> _tmap.VectorUint


                    Create a MinHash vector from a sparse binary array.

        Arguments:
                        vec (:obj:`list`): A Python list containing indices of ones in a binary array

        Returns:
                        :obj:`VectorUint`: A MinHash vector


        2. from_sparse_binary_array(self: _tmap.Minhash, arg0: _tmap.VectorUint) -> _tmap.VectorUint


                    Create a MinHash vector from a sparse binary array.

        Arguments:
                        vec (:obj:`VectorUint`): A Python list containing indices of ones in a binary array

        Returns:
                        :obj:`VectorUint`: A MinHash vector

        """

    @overload
    def from_sparse_binary_array(self, arg0: VectorUint) -> VectorUint:
        """from_sparse_binary_array(*args, **kwargs)
        Overloaded function.

        1. from_sparse_binary_array(self: _tmap.Minhash, arg0: list) -> _tmap.VectorUint


                    Create a MinHash vector from a sparse binary array.

        Arguments:
                        vec (:obj:`list`): A Python list containing indices of ones in a binary array

        Returns:
                        :obj:`VectorUint`: A MinHash vector


        2. from_sparse_binary_array(self: _tmap.Minhash, arg0: _tmap.VectorUint) -> _tmap.VectorUint


                    Create a MinHash vector from a sparse binary array.

        Arguments:
                        vec (:obj:`VectorUint`): A Python list containing indices of ones in a binary array

        Returns:
                        :obj:`VectorUint`: A MinHash vector

        """

    def from_string_array(self, arg0: list[str]) -> VectorUint:
        """from_string_array(self: _tmap.Minhash, arg0: list[str]) -> _tmap.VectorUint.

        Create a MinHash vector from a string array.

        Arguments:
            vec (:obj:`list` of :obj:`str`): A vector containing strings

        Returns:
            :obj:`VectorUint`: A MinHash vector

        """

    @overload
    def from_weight_array(self, vec: list, method: str = ...) -> VectorUint:
        """from_weight_array(*args, **kwargs)
        Overloaded function.

        1. from_weight_array(self: _tmap.Minhash, vec: list, method: str = 'ICWS') -> _tmap.VectorUint


                    Create a MinHash vector from a :obj:`list`.

        Arguments:
                        vec (:obj:`list`): A Python list containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`VectorUint`: A MinHash vector


        2. from_weight_array(self: _tmap.Minhash, vec: _tmap.VectorFloat, method: str = 'ICWS') -> _tmap.VectorUint


                    Create a MinHash vector from a :obj:`float` array.

        Arguments:
                        vec (:obj:`VectorFloat`): A vector containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`VectorUint`: A MinHash vector

        """

    @overload
    def from_weight_array(self, vec: VectorFloat, method: str = ...) -> VectorUint:
        """from_weight_array(*args, **kwargs)
        Overloaded function.

        1. from_weight_array(self: _tmap.Minhash, vec: list, method: str = 'ICWS') -> _tmap.VectorUint


                    Create a MinHash vector from a :obj:`list`.

        Arguments:
                        vec (:obj:`list`): A Python list containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`VectorUint`: A MinHash vector


        2. from_weight_array(self: _tmap.Minhash, vec: _tmap.VectorFloat, method: str = 'ICWS') -> _tmap.VectorUint


                    Create a MinHash vector from a :obj:`float` array.

        Arguments:
                        vec (:obj:`VectorFloat`): A vector containing :obj:`float` values

        Keyword Arguments:
                        method (:obj:`str`): The weighted hashing method to use (ICWS or I2CWS)

        Returns:
                        :obj:`VectorUint`: A MinHash vector

        """

    def get_distance(self, arg0: VectorUint, arg1: VectorUint) -> float:
        """get_distance(self: _tmap.Minhash, arg0: _tmap.VectorUint, arg1: _tmap.VectorUint) -> float.

        Calculate the Jaccard distance between two MinHash vectors.

        Arguments:
            vec_a (:obj:`VectorUint`): A MinHash vector
            vec_b (:obj:`VectorUint`): A MinHash vector

        Returns:
            :obj:`float` The Jaccard distance

        """

    def get_weighted_distance(self, arg0: VectorUint, arg1: VectorUint) -> float:
        """get_weighted_distance(self: _tmap.Minhash, arg0: _tmap.VectorUint, arg1: _tmap.VectorUint) -> float.

        Calculate the weighted Jaccard distance between two MinHash vectors.

        Arguments:
            vec_a (:obj:`VectorUint`): A weighted MinHash vector
            vec_b (:obj:`VectorUint`): A weighted MinHash vector

        Returns:
            :obj:`float` The Jaccard distance

        """

class Placer:
    __members__: ClassVar[dict] = ...  # read-only
    Barycenter: ClassVar[Placer] = ...
    Circle: ClassVar[Placer] = ...
    Median: ClassVar[Placer] = ...
    Random: ClassVar[Placer] = ...
    Solar: ClassVar[Placer] = ...
    Zero: ClassVar[Placer] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: _tmap.Placer, value: int) -> None."""

    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool."""

    def __ge__(self, other: object) -> bool:
        """__ge__(self: object, other: object) -> bool."""

    def __gt__(self, other: object) -> bool:
        """__gt__(self: object, other: object) -> bool."""

    def __hash__(self) -> int:
        """__hash__(self: object) -> int."""

    def __index__(self) -> int:
        """__index__(self: _tmap.Placer) -> int."""

    def __int__(self) -> int:
        """__int__(self: _tmap.Placer) -> int."""

    def __le__(self, other: object) -> bool:
        """__le__(self: object, other: object) -> bool."""

    def __lt__(self, other: object) -> bool:
        """__lt__(self: object, other: object) -> bool."""

    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool."""

    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ScalingType:
    __members__: ClassVar[dict] = ...  # read-only
    Absolute: ClassVar[ScalingType] = ...
    RelativeToAvgLength: ClassVar[ScalingType] = ...
    RelativeToDesiredLength: ClassVar[ScalingType] = ...
    RelativeToDrawing: ClassVar[ScalingType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: _tmap.ScalingType, value: int) -> None."""

    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool."""

    def __ge__(self, other: object) -> bool:
        """__ge__(self: object, other: object) -> bool."""

    def __gt__(self, other: object) -> bool:
        """__gt__(self: object, other: object) -> bool."""

    def __hash__(self) -> int:
        """__hash__(self: object) -> int."""

    def __index__(self) -> int:
        """__index__(self: _tmap.ScalingType) -> int."""

    def __int__(self) -> int:
        """__int__(self: _tmap.ScalingType) -> int."""

    def __le__(self, other: object) -> bool:
        """__le__(self: object, other: object) -> bool."""

    def __lt__(self, other: object) -> bool:
        """__lt__(self: object, other: object) -> bool."""

    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool."""

    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class TestSub:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class VectorFloat:
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorFloat) -> None

        2. __init__(self: _tmap.VectorFloat, arg0: _tmap.VectorFloat) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorFloat, arg0: Iterable) -> None
        """

    @overload
    def __init__(self, arg0: VectorFloat) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorFloat) -> None

        2. __init__(self: _tmap.VectorFloat, arg0: _tmap.VectorFloat) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorFloat, arg0: Iterable) -> None
        """

    @overload
    def __init__(self, arg0: Iterable) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorFloat) -> None

        2. __init__(self: _tmap.VectorFloat, arg0: _tmap.VectorFloat) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorFloat, arg0: Iterable) -> None
        """

    def append(self, x: float) -> None:
        """append(self: _tmap.VectorFloat, x: float) -> None.

        Add an item to the end of the list
        """

    def clear(self) -> None:
        """clear(self: _tmap.VectorFloat) -> None.

        Clear the contents
        """

    def count(self, x: float) -> int:
        """count(self: _tmap.VectorFloat, x: float) -> int.

        Return the number of times ``x`` appears in the list
        """

    @overload
    def extend(self, L: VectorFloat) -> None:
        """extend(*args, **kwargs)
        Overloaded function.

        1. extend(self: _tmap.VectorFloat, L: _tmap.VectorFloat) -> None

        Extend the list by appending all the items in the given list

        2. extend(self: _tmap.VectorFloat, L: Iterable) -> None

        Extend the list by appending all the items in the given list
        """

    @overload
    def extend(self, L: Iterable) -> None:
        """extend(*args, **kwargs)
        Overloaded function.

        1. extend(self: _tmap.VectorFloat, L: _tmap.VectorFloat) -> None

        Extend the list by appending all the items in the given list

        2. extend(self: _tmap.VectorFloat, L: Iterable) -> None

        Extend the list by appending all the items in the given list
        """

    def insert(self, i: int, x: float) -> None:
        """insert(self: _tmap.VectorFloat, i: int, x: float) -> None.

        Insert an item at a given position.
        """

    @overload
    def pop(self) -> float:
        """pop(*args, **kwargs)
        Overloaded function.

        1. pop(self: _tmap.VectorFloat) -> float

        Remove and return the last item

        2. pop(self: _tmap.VectorFloat, i: int) -> float

        Remove and return the item at index ``i``
        """

    @overload
    def pop(self, i: int) -> float:
        """pop(*args, **kwargs)
        Overloaded function.

        1. pop(self: _tmap.VectorFloat) -> float

        Remove and return the last item

        2. pop(self: _tmap.VectorFloat, i: int) -> float

        Remove and return the item at index ``i``
        """

    def remove(self, x: float) -> None:
        """remove(self: _tmap.VectorFloat, x: float) -> None.

        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """

    def __bool__(self) -> bool:
        """__bool__(self: _tmap.VectorFloat) -> bool.

        Check whether the list is nonempty
        """

    def __contains__(self, x: float) -> bool:
        """__contains__(self: _tmap.VectorFloat, x: float) -> bool.

        Return true the container contains ``x``
        """

    @overload
    def __delitem__(self, arg0: int) -> None:
        """__delitem__(*args, **kwargs)
        Overloaded function.

        1. __delitem__(self: _tmap.VectorFloat, arg0: int) -> None

        Delete the list elements at index ``i``

        2. __delitem__(self: _tmap.VectorFloat, arg0: slice) -> None

        Delete list elements using a slice object
        """

    @overload
    def __delitem__(self, arg0: slice) -> None:
        """__delitem__(*args, **kwargs)
        Overloaded function.

        1. __delitem__(self: _tmap.VectorFloat, arg0: int) -> None

        Delete the list elements at index ``i``

        2. __delitem__(self: _tmap.VectorFloat, arg0: slice) -> None

        Delete list elements using a slice object
        """

    def __eq__(self, arg0: VectorFloat) -> bool:
        """__eq__(self: _tmap.VectorFloat, arg0: _tmap.VectorFloat) -> bool."""

    @overload
    def __getitem__(self, s: slice) -> VectorFloat:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: _tmap.VectorFloat, s: slice) -> _tmap.VectorFloat

        Retrieve list elements using a slice object

        2. __getitem__(self: _tmap.VectorFloat, arg0: int) -> float
        """

    @overload
    def __getitem__(self, arg0: int) -> float:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: _tmap.VectorFloat, s: slice) -> _tmap.VectorFloat

        Retrieve list elements using a slice object

        2. __getitem__(self: _tmap.VectorFloat, arg0: int) -> float
        """

    def __iter__(self) -> Iterator:
        """__iter__(self: _tmap.VectorFloat) -> Iterator."""

    def __len__(self) -> int:
        """__len__(self: _tmap.VectorFloat) -> int."""

    def __ne__(self, arg0: VectorFloat) -> bool:
        """__ne__(self: _tmap.VectorFloat, arg0: _tmap.VectorFloat) -> bool."""

    @overload
    def __setitem__(self, arg0: int, arg1: float) -> None:
        """__setitem__(*args, **kwargs)
        Overloaded function.

        1. __setitem__(self: _tmap.VectorFloat, arg0: int, arg1: float) -> None

        2. __setitem__(self: _tmap.VectorFloat, arg0: slice, arg1: _tmap.VectorFloat) -> None

        Assign list elements using a slice object
        """

    @overload
    def __setitem__(self, arg0: slice, arg1: VectorFloat) -> None:
        """__setitem__(*args, **kwargs)
        Overloaded function.

        1. __setitem__(self: _tmap.VectorFloat, arg0: int, arg1: float) -> None

        2. __setitem__(self: _tmap.VectorFloat, arg0: slice, arg1: _tmap.VectorFloat) -> None

        Assign list elements using a slice object
        """

class VectorUchar:
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorUchar) -> None

        2. __init__(self: _tmap.VectorUchar, arg0: _tmap.VectorUchar) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorUchar, arg0: Iterable) -> None
        """

    @overload
    def __init__(self, arg0: VectorUchar) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorUchar) -> None

        2. __init__(self: _tmap.VectorUchar, arg0: _tmap.VectorUchar) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorUchar, arg0: Iterable) -> None
        """

    @overload
    def __init__(self, arg0: Iterable) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorUchar) -> None

        2. __init__(self: _tmap.VectorUchar, arg0: _tmap.VectorUchar) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorUchar, arg0: Iterable) -> None
        """

    def append(self, x: int) -> None:
        """append(self: _tmap.VectorUchar, x: int) -> None.

        Add an item to the end of the list
        """

    def clear(self) -> None:
        """clear(self: _tmap.VectorUchar) -> None.

        Clear the contents
        """

    def count(self, x: int) -> int:
        """count(self: _tmap.VectorUchar, x: int) -> int.

        Return the number of times ``x`` appears in the list
        """

    @overload
    def extend(self, L: VectorUchar) -> None:
        """extend(*args, **kwargs)
        Overloaded function.

        1. extend(self: _tmap.VectorUchar, L: _tmap.VectorUchar) -> None

        Extend the list by appending all the items in the given list

        2. extend(self: _tmap.VectorUchar, L: Iterable) -> None

        Extend the list by appending all the items in the given list
        """

    @overload
    def extend(self, L: Iterable) -> None:
        """extend(*args, **kwargs)
        Overloaded function.

        1. extend(self: _tmap.VectorUchar, L: _tmap.VectorUchar) -> None

        Extend the list by appending all the items in the given list

        2. extend(self: _tmap.VectorUchar, L: Iterable) -> None

        Extend the list by appending all the items in the given list
        """

    def insert(self, i: int, x: int) -> None:
        """insert(self: _tmap.VectorUchar, i: int, x: int) -> None.

        Insert an item at a given position.
        """

    @overload
    def pop(self) -> int:
        """pop(*args, **kwargs)
        Overloaded function.

        1. pop(self: _tmap.VectorUchar) -> int

        Remove and return the last item

        2. pop(self: _tmap.VectorUchar, i: int) -> int

        Remove and return the item at index ``i``
        """

    @overload
    def pop(self, i: int) -> int:
        """pop(*args, **kwargs)
        Overloaded function.

        1. pop(self: _tmap.VectorUchar) -> int

        Remove and return the last item

        2. pop(self: _tmap.VectorUchar, i: int) -> int

        Remove and return the item at index ``i``
        """

    def remove(self, x: int) -> None:
        """remove(self: _tmap.VectorUchar, x: int) -> None.

        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """

    def __bool__(self) -> bool:
        """__bool__(self: _tmap.VectorUchar) -> bool.

        Check whether the list is nonempty
        """

    def __contains__(self, x: int) -> bool:
        """__contains__(self: _tmap.VectorUchar, x: int) -> bool.

        Return true the container contains ``x``
        """

    @overload
    def __delitem__(self, arg0: int) -> None:
        """__delitem__(*args, **kwargs)
        Overloaded function.

        1. __delitem__(self: _tmap.VectorUchar, arg0: int) -> None

        Delete the list elements at index ``i``

        2. __delitem__(self: _tmap.VectorUchar, arg0: slice) -> None

        Delete list elements using a slice object
        """

    @overload
    def __delitem__(self, arg0: slice) -> None:
        """__delitem__(*args, **kwargs)
        Overloaded function.

        1. __delitem__(self: _tmap.VectorUchar, arg0: int) -> None

        Delete the list elements at index ``i``

        2. __delitem__(self: _tmap.VectorUchar, arg0: slice) -> None

        Delete list elements using a slice object
        """

    def __eq__(self, arg0: VectorUchar) -> bool:
        """__eq__(self: _tmap.VectorUchar, arg0: _tmap.VectorUchar) -> bool."""

    @overload
    def __getitem__(self, s: slice) -> VectorUchar:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: _tmap.VectorUchar, s: slice) -> _tmap.VectorUchar

        Retrieve list elements using a slice object

        2. __getitem__(self: _tmap.VectorUchar, arg0: int) -> int
        """

    @overload
    def __getitem__(self, arg0: int) -> int:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: _tmap.VectorUchar, s: slice) -> _tmap.VectorUchar

        Retrieve list elements using a slice object

        2. __getitem__(self: _tmap.VectorUchar, arg0: int) -> int
        """

    def __iter__(self) -> Iterator:
        """__iter__(self: _tmap.VectorUchar) -> Iterator."""

    def __len__(self) -> int:
        """__len__(self: _tmap.VectorUchar) -> int."""

    def __ne__(self, arg0: VectorUchar) -> bool:
        """__ne__(self: _tmap.VectorUchar, arg0: _tmap.VectorUchar) -> bool."""

    @overload
    def __setitem__(self, arg0: int, arg1: int) -> None:
        """__setitem__(*args, **kwargs)
        Overloaded function.

        1. __setitem__(self: _tmap.VectorUchar, arg0: int, arg1: int) -> None

        2. __setitem__(self: _tmap.VectorUchar, arg0: slice, arg1: _tmap.VectorUchar) -> None

        Assign list elements using a slice object
        """

    @overload
    def __setitem__(self, arg0: slice, arg1: VectorUchar) -> None:
        """__setitem__(*args, **kwargs)
        Overloaded function.

        1. __setitem__(self: _tmap.VectorUchar, arg0: int, arg1: int) -> None

        2. __setitem__(self: _tmap.VectorUchar, arg0: slice, arg1: _tmap.VectorUchar) -> None

        Assign list elements using a slice object
        """

class VectorUint:
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorUint) -> None

        2. __init__(self: _tmap.VectorUint, arg0: _tmap.VectorUint) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorUint, arg0: Iterable) -> None
        """

    @overload
    def __init__(self, arg0: VectorUint) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorUint) -> None

        2. __init__(self: _tmap.VectorUint, arg0: _tmap.VectorUint) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorUint, arg0: Iterable) -> None
        """

    @overload
    def __init__(self, arg0: Iterable) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorUint) -> None

        2. __init__(self: _tmap.VectorUint, arg0: _tmap.VectorUint) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorUint, arg0: Iterable) -> None
        """

    def append(self, x: int) -> None:
        """append(self: _tmap.VectorUint, x: int) -> None.

        Add an item to the end of the list
        """

    def clear(self) -> None:
        """clear(self: _tmap.VectorUint) -> None.

        Clear the contents
        """

    def count(self, x: int) -> int:
        """count(self: _tmap.VectorUint, x: int) -> int.

        Return the number of times ``x`` appears in the list
        """

    @overload
    def extend(self, L: VectorUint) -> None:
        """extend(*args, **kwargs)
        Overloaded function.

        1. extend(self: _tmap.VectorUint, L: _tmap.VectorUint) -> None

        Extend the list by appending all the items in the given list

        2. extend(self: _tmap.VectorUint, L: Iterable) -> None

        Extend the list by appending all the items in the given list
        """

    @overload
    def extend(self, L: Iterable) -> None:
        """extend(*args, **kwargs)
        Overloaded function.

        1. extend(self: _tmap.VectorUint, L: _tmap.VectorUint) -> None

        Extend the list by appending all the items in the given list

        2. extend(self: _tmap.VectorUint, L: Iterable) -> None

        Extend the list by appending all the items in the given list
        """

    def insert(self, i: int, x: int) -> None:
        """insert(self: _tmap.VectorUint, i: int, x: int) -> None.

        Insert an item at a given position.
        """

    @overload
    def pop(self) -> int:
        """pop(*args, **kwargs)
        Overloaded function.

        1. pop(self: _tmap.VectorUint) -> int

        Remove and return the last item

        2. pop(self: _tmap.VectorUint, i: int) -> int

        Remove and return the item at index ``i``
        """

    @overload
    def pop(self, i: int) -> int:
        """pop(*args, **kwargs)
        Overloaded function.

        1. pop(self: _tmap.VectorUint) -> int

        Remove and return the last item

        2. pop(self: _tmap.VectorUint, i: int) -> int

        Remove and return the item at index ``i``
        """

    def remove(self, x: int) -> None:
        """remove(self: _tmap.VectorUint, x: int) -> None.

        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """

    def __bool__(self) -> bool:
        """__bool__(self: _tmap.VectorUint) -> bool.

        Check whether the list is nonempty
        """

    def __contains__(self, x: int) -> bool:
        """__contains__(self: _tmap.VectorUint, x: int) -> bool.

        Return true the container contains ``x``
        """

    @overload
    def __delitem__(self, arg0: int) -> None:
        """__delitem__(*args, **kwargs)
        Overloaded function.

        1. __delitem__(self: _tmap.VectorUint, arg0: int) -> None

        Delete the list elements at index ``i``

        2. __delitem__(self: _tmap.VectorUint, arg0: slice) -> None

        Delete list elements using a slice object
        """

    @overload
    def __delitem__(self, arg0: slice) -> None:
        """__delitem__(*args, **kwargs)
        Overloaded function.

        1. __delitem__(self: _tmap.VectorUint, arg0: int) -> None

        Delete the list elements at index ``i``

        2. __delitem__(self: _tmap.VectorUint, arg0: slice) -> None

        Delete list elements using a slice object
        """

    def __eq__(self, arg0: VectorUint) -> bool:
        """__eq__(self: _tmap.VectorUint, arg0: _tmap.VectorUint) -> bool."""

    @overload
    def __getitem__(self, s: slice) -> VectorUint:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: _tmap.VectorUint, s: slice) -> _tmap.VectorUint

        Retrieve list elements using a slice object

        2. __getitem__(self: _tmap.VectorUint, arg0: int) -> int
        """

    @overload
    def __getitem__(self, arg0: int) -> int:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: _tmap.VectorUint, s: slice) -> _tmap.VectorUint

        Retrieve list elements using a slice object

        2. __getitem__(self: _tmap.VectorUint, arg0: int) -> int
        """

    def __iter__(self) -> Iterator:
        """__iter__(self: _tmap.VectorUint) -> Iterator."""

    def __len__(self) -> int:
        """__len__(self: _tmap.VectorUint) -> int."""

    def __ne__(self, arg0: VectorUint) -> bool:
        """__ne__(self: _tmap.VectorUint, arg0: _tmap.VectorUint) -> bool."""

    @overload
    def __setitem__(self, arg0: int, arg1: int) -> None:
        """__setitem__(*args, **kwargs)
        Overloaded function.

        1. __setitem__(self: _tmap.VectorUint, arg0: int, arg1: int) -> None

        2. __setitem__(self: _tmap.VectorUint, arg0: slice, arg1: _tmap.VectorUint) -> None

        Assign list elements using a slice object
        """

    @overload
    def __setitem__(self, arg0: slice, arg1: VectorUint) -> None:
        """__setitem__(*args, **kwargs)
        Overloaded function.

        1. __setitem__(self: _tmap.VectorUint, arg0: int, arg1: int) -> None

        2. __setitem__(self: _tmap.VectorUint, arg0: slice, arg1: _tmap.VectorUint) -> None

        Assign list elements using a slice object
        """

class VectorUlong:
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorUlong) -> None

        2. __init__(self: _tmap.VectorUlong, arg0: _tmap.VectorUlong) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorUlong, arg0: Iterable) -> None
        """

    @overload
    def __init__(self, arg0: VectorUlong) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorUlong) -> None

        2. __init__(self: _tmap.VectorUlong, arg0: _tmap.VectorUlong) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorUlong, arg0: Iterable) -> None
        """

    @overload
    def __init__(self, arg0: Iterable) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorUlong) -> None

        2. __init__(self: _tmap.VectorUlong, arg0: _tmap.VectorUlong) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorUlong, arg0: Iterable) -> None
        """

    def append(self, x: int) -> None:
        """append(self: _tmap.VectorUlong, x: int) -> None.

        Add an item to the end of the list
        """

    def clear(self) -> None:
        """clear(self: _tmap.VectorUlong) -> None.

        Clear the contents
        """

    def count(self, x: int) -> int:
        """count(self: _tmap.VectorUlong, x: int) -> int.

        Return the number of times ``x`` appears in the list
        """

    @overload
    def extend(self, L: VectorUlong) -> None:
        """extend(*args, **kwargs)
        Overloaded function.

        1. extend(self: _tmap.VectorUlong, L: _tmap.VectorUlong) -> None

        Extend the list by appending all the items in the given list

        2. extend(self: _tmap.VectorUlong, L: Iterable) -> None

        Extend the list by appending all the items in the given list
        """

    @overload
    def extend(self, L: Iterable) -> None:
        """extend(*args, **kwargs)
        Overloaded function.

        1. extend(self: _tmap.VectorUlong, L: _tmap.VectorUlong) -> None

        Extend the list by appending all the items in the given list

        2. extend(self: _tmap.VectorUlong, L: Iterable) -> None

        Extend the list by appending all the items in the given list
        """

    def insert(self, i: int, x: int) -> None:
        """insert(self: _tmap.VectorUlong, i: int, x: int) -> None.

        Insert an item at a given position.
        """

    @overload
    def pop(self) -> int:
        """pop(*args, **kwargs)
        Overloaded function.

        1. pop(self: _tmap.VectorUlong) -> int

        Remove and return the last item

        2. pop(self: _tmap.VectorUlong, i: int) -> int

        Remove and return the item at index ``i``
        """

    @overload
    def pop(self, i: int) -> int:
        """pop(*args, **kwargs)
        Overloaded function.

        1. pop(self: _tmap.VectorUlong) -> int

        Remove and return the last item

        2. pop(self: _tmap.VectorUlong, i: int) -> int

        Remove and return the item at index ``i``
        """

    def remove(self, x: int) -> None:
        """remove(self: _tmap.VectorUlong, x: int) -> None.

        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """

    def __bool__(self) -> bool:
        """__bool__(self: _tmap.VectorUlong) -> bool.

        Check whether the list is nonempty
        """

    def __contains__(self, x: int) -> bool:
        """__contains__(self: _tmap.VectorUlong, x: int) -> bool.

        Return true the container contains ``x``
        """

    @overload
    def __delitem__(self, arg0: int) -> None:
        """__delitem__(*args, **kwargs)
        Overloaded function.

        1. __delitem__(self: _tmap.VectorUlong, arg0: int) -> None

        Delete the list elements at index ``i``

        2. __delitem__(self: _tmap.VectorUlong, arg0: slice) -> None

        Delete list elements using a slice object
        """

    @overload
    def __delitem__(self, arg0: slice) -> None:
        """__delitem__(*args, **kwargs)
        Overloaded function.

        1. __delitem__(self: _tmap.VectorUlong, arg0: int) -> None

        Delete the list elements at index ``i``

        2. __delitem__(self: _tmap.VectorUlong, arg0: slice) -> None

        Delete list elements using a slice object
        """

    def __eq__(self, arg0: VectorUlong) -> bool:
        """__eq__(self: _tmap.VectorUlong, arg0: _tmap.VectorUlong) -> bool."""

    @overload
    def __getitem__(self, s: slice) -> VectorUlong:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: _tmap.VectorUlong, s: slice) -> _tmap.VectorUlong

        Retrieve list elements using a slice object

        2. __getitem__(self: _tmap.VectorUlong, arg0: int) -> int
        """

    @overload
    def __getitem__(self, arg0: int) -> int:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: _tmap.VectorUlong, s: slice) -> _tmap.VectorUlong

        Retrieve list elements using a slice object

        2. __getitem__(self: _tmap.VectorUlong, arg0: int) -> int
        """

    def __iter__(self) -> Iterator:
        """__iter__(self: _tmap.VectorUlong) -> Iterator."""

    def __len__(self) -> int:
        """__len__(self: _tmap.VectorUlong) -> int."""

    def __ne__(self, arg0: VectorUlong) -> bool:
        """__ne__(self: _tmap.VectorUlong, arg0: _tmap.VectorUlong) -> bool."""

    @overload
    def __setitem__(self, arg0: int, arg1: int) -> None:
        """__setitem__(*args, **kwargs)
        Overloaded function.

        1. __setitem__(self: _tmap.VectorUlong, arg0: int, arg1: int) -> None

        2. __setitem__(self: _tmap.VectorUlong, arg0: slice, arg1: _tmap.VectorUlong) -> None

        Assign list elements using a slice object
        """

    @overload
    def __setitem__(self, arg0: slice, arg1: VectorUlong) -> None:
        """__setitem__(*args, **kwargs)
        Overloaded function.

        1. __setitem__(self: _tmap.VectorUlong, arg0: int, arg1: int) -> None

        2. __setitem__(self: _tmap.VectorUlong, arg0: slice, arg1: _tmap.VectorUlong) -> None

        Assign list elements using a slice object
        """

class VectorUsmall:
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorUsmall) -> None

        2. __init__(self: _tmap.VectorUsmall, arg0: _tmap.VectorUsmall) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorUsmall, arg0: Iterable) -> None
        """

    @overload
    def __init__(self, arg0: VectorUsmall) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorUsmall) -> None

        2. __init__(self: _tmap.VectorUsmall, arg0: _tmap.VectorUsmall) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorUsmall, arg0: Iterable) -> None
        """

    @overload
    def __init__(self, arg0: Iterable) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _tmap.VectorUsmall) -> None

        2. __init__(self: _tmap.VectorUsmall, arg0: _tmap.VectorUsmall) -> None

        Copy constructor

        3. __init__(self: _tmap.VectorUsmall, arg0: Iterable) -> None
        """

    def append(self, x: int) -> None:
        """append(self: _tmap.VectorUsmall, x: int) -> None.

        Add an item to the end of the list
        """

    def clear(self) -> None:
        """clear(self: _tmap.VectorUsmall) -> None.

        Clear the contents
        """

    def count(self, x: int) -> int:
        """count(self: _tmap.VectorUsmall, x: int) -> int.

        Return the number of times ``x`` appears in the list
        """

    @overload
    def extend(self, L: VectorUsmall) -> None:
        """extend(*args, **kwargs)
        Overloaded function.

        1. extend(self: _tmap.VectorUsmall, L: _tmap.VectorUsmall) -> None

        Extend the list by appending all the items in the given list

        2. extend(self: _tmap.VectorUsmall, L: Iterable) -> None

        Extend the list by appending all the items in the given list
        """

    @overload
    def extend(self, L: Iterable) -> None:
        """extend(*args, **kwargs)
        Overloaded function.

        1. extend(self: _tmap.VectorUsmall, L: _tmap.VectorUsmall) -> None

        Extend the list by appending all the items in the given list

        2. extend(self: _tmap.VectorUsmall, L: Iterable) -> None

        Extend the list by appending all the items in the given list
        """

    def insert(self, i: int, x: int) -> None:
        """insert(self: _tmap.VectorUsmall, i: int, x: int) -> None.

        Insert an item at a given position.
        """

    @overload
    def pop(self) -> int:
        """pop(*args, **kwargs)
        Overloaded function.

        1. pop(self: _tmap.VectorUsmall) -> int

        Remove and return the last item

        2. pop(self: _tmap.VectorUsmall, i: int) -> int

        Remove and return the item at index ``i``
        """

    @overload
    def pop(self, i: int) -> int:
        """pop(*args, **kwargs)
        Overloaded function.

        1. pop(self: _tmap.VectorUsmall) -> int

        Remove and return the last item

        2. pop(self: _tmap.VectorUsmall, i: int) -> int

        Remove and return the item at index ``i``
        """

    def remove(self, x: int) -> None:
        """remove(self: _tmap.VectorUsmall, x: int) -> None.

        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """

    def __bool__(self) -> bool:
        """__bool__(self: _tmap.VectorUsmall) -> bool.

        Check whether the list is nonempty
        """

    def __contains__(self, x: int) -> bool:
        """__contains__(self: _tmap.VectorUsmall, x: int) -> bool.

        Return true the container contains ``x``
        """

    @overload
    def __delitem__(self, arg0: int) -> None:
        """__delitem__(*args, **kwargs)
        Overloaded function.

        1. __delitem__(self: _tmap.VectorUsmall, arg0: int) -> None

        Delete the list elements at index ``i``

        2. __delitem__(self: _tmap.VectorUsmall, arg0: slice) -> None

        Delete list elements using a slice object
        """

    @overload
    def __delitem__(self, arg0: slice) -> None:
        """__delitem__(*args, **kwargs)
        Overloaded function.

        1. __delitem__(self: _tmap.VectorUsmall, arg0: int) -> None

        Delete the list elements at index ``i``

        2. __delitem__(self: _tmap.VectorUsmall, arg0: slice) -> None

        Delete list elements using a slice object
        """

    def __eq__(self, arg0: VectorUsmall) -> bool:
        """__eq__(self: _tmap.VectorUsmall, arg0: _tmap.VectorUsmall) -> bool."""

    @overload
    def __getitem__(self, s: slice) -> VectorUsmall:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: _tmap.VectorUsmall, s: slice) -> _tmap.VectorUsmall

        Retrieve list elements using a slice object

        2. __getitem__(self: _tmap.VectorUsmall, arg0: int) -> int
        """

    @overload
    def __getitem__(self, arg0: int) -> int:
        """__getitem__(*args, **kwargs)
        Overloaded function.

        1. __getitem__(self: _tmap.VectorUsmall, s: slice) -> _tmap.VectorUsmall

        Retrieve list elements using a slice object

        2. __getitem__(self: _tmap.VectorUsmall, arg0: int) -> int
        """

    def __iter__(self) -> Iterator:
        """__iter__(self: _tmap.VectorUsmall) -> Iterator."""

    def __len__(self) -> int:
        """__len__(self: _tmap.VectorUsmall) -> int."""

    def __ne__(self, arg0: VectorUsmall) -> bool:
        """__ne__(self: _tmap.VectorUsmall, arg0: _tmap.VectorUsmall) -> bool."""

    @overload
    def __setitem__(self, arg0: int, arg1: int) -> None:
        """__setitem__(*args, **kwargs)
        Overloaded function.

        1. __setitem__(self: _tmap.VectorUsmall, arg0: int, arg1: int) -> None

        2. __setitem__(self: _tmap.VectorUsmall, arg0: slice, arg1: _tmap.VectorUsmall) -> None

        Assign list elements using a slice object
        """

    @overload
    def __setitem__(self, arg0: slice, arg1: VectorUsmall) -> None:
        """__setitem__(*args, **kwargs)
        Overloaded function.

        1. __setitem__(self: _tmap.VectorUsmall, arg0: int, arg1: int) -> None

        2. __setitem__(self: _tmap.VectorUsmall, arg0: slice, arg1: _tmap.VectorUsmall) -> None

        Assign list elements using a slice object
        """

def MSDR(gp: GraphProperties) -> list[VectorUint]:
    """MSDR(gp: _tmap.GraphProperties) -> list[_tmap.VectorUint].

    brief Implementation of the MSDR clustering algorithm.

    Arguments:
        gp (:obj:`GraphProperties`): A GraphProperties object

    Returns:
      :obj:`VectorUint[VectorUint]` The vertex ids divided in clusters

    """

def get_clusters(
    gp: GraphProperties, classes: VectorUint
) -> list[tuple[int, VectorUint]]:
    """get_clusters(gp: _tmap.GraphProperties, classes: _tmap.VectorUint) -> list[tuple[int, _tmap.VectorUint]].

    brief Creates clusters from a minimum spanning tree.

    Arguments:
        gp (:obj:`GraphProperties`): A GraphProperties object
        classes (:obj:`VectorUint`): The classes of the vertices

    Returns:
      :obj:`float` The Jaccard distance

    """

def get_topological_distances(gp: GraphProperties, v: int) -> VectorUint:
    """get_topological_distances(gp: _tmap.GraphProperties, v: int) -> _tmap.VectorUint.

    brief Gets the topological distances of a vertex to all other vertices.

    Arguments:
        gp (:obj:`VectorFloat`): A GraphProperties object
        v (:obj:`int`): The vertex id of the node to be analysed

    Returns:
        :obj:`list[int]`: The topological distances to vertex v.

    """

def layout_from_edge_list(
    vertex_count: int,
    edges: list[tuple[int, int, float]],
    config: LayoutConfiguration = ...,
    keep_knn: bool = ...,
    create_mst: bool = ...,
) -> tuple[VectorFloat, VectorFloat, VectorUint, VectorUint, GraphProperties]:
    """layout_from_edge_list(vertex_count: int, edges: list[tuple[int, int, float]], config: _tmap.LayoutConfiguration = k: 10
    kc: 10
    fme_iterations: 1000
    fme_randomize: 0
    fme_threads: 4
    fme_precision: 4
    sl_repeats: 1
    sl_extra_scaling_steps: 2
    sl_scaling_x: 1.000000
    sl_scaling_y: 1.000000
    sl_scaling_type: RelativeToDrawing
    mmm_repeats: 1
    placer: Barycenter
    merger: LocalBiconnected
    merger_factor: 2.000000
    merger_adjustment: 0
    node_size0.015385, keep_knn: bool = False, create_mst: bool = True) -> tuple[_tmap.VectorFloat, _tmap.VectorFloat, _tmap.VectorUint, _tmap.VectorUint, _tmap.GraphProperties].


            Create minimum spanning tree or k-nearest neighbor graph coordinates and topology from an edge list.

    Arguments:
                vertex_count (:obj:`int`): The number of vertices in the edge list
                edges (:obj:`list` of :obj:`tuple[int, int, float]`): An edge list defining a graph

    Keyword Arguments:
                config (:obj:`LayoutConfiguration`, optional): An :obj:`LayoutConfiguration` instance
                create_mst (:obj:`bool`): Whether to create a minimum spanning tree or to return coordinates and topology for the k-nearest neighbor graph

    Returns:
                :obj:`tuple[VectorFloat, VectorFloat, VectorUint, VectorUint, GraphProperties]`: The x and y coordinates of the vertices, the ids of the vertices spanning the edges, and information on the graph

    """

def layout_from_edge_list_native(
    vertex_count: int,
    edges: list[tuple[int, int, float]],
    config: LayoutConfiguration = ...,
    keep_knn: bool = ...,
    create_mst: bool = ...,
) -> tuple:
    """layout_from_edge_list_native(vertex_count: int, edges: list[tuple[int, int, float]], config: _tmap.LayoutConfiguration = k: 10
    kc: 10
    fme_iterations: 1000
    fme_randomize: 0
    fme_threads: 4
    fme_precision: 4
    sl_repeats: 1
    sl_extra_scaling_steps: 2
    sl_scaling_x: 1.000000
    sl_scaling_y: 1.000000
    sl_scaling_type: RelativeToDrawing
    mmm_repeats: 1
    placer: Barycenter
    merger: LocalBiconnected
    merger_factor: 2.000000
    merger_adjustment: 0
    node_size0.015385, keep_knn: bool = False, create_mst: bool = True) -> tuple.


            Create minimum spanning tree or k-nearest neighbor graph coordinates and topology from an edge list. This method returns native python lists and objects.

    Arguments:
                vertex_count (:obj:`int`): The number of vertices in the edge list
                edges (:obj:`list` of :obj:`tuple[int, int, float]`): An edge list defining a graph

    Keyword Arguments:
                config (:obj:`LayoutConfiguration`, optional): An :obj:`LayoutConfiguration` instance
                create_mst (:obj:`bool`): Whether to create a minimum spanning tree or to return coordinates and topology for the k-nearest neighbor graph

    Returns:
                :obj:`tuple[VectorFloat, VectorFloat, VectorUint, VectorUint, GraphProperties]`: The x and y coordinates of the vertices, the ids of the vertices spanning the edges, and information on the graph

    """

def layout_from_lsh_forest(
    lsh_forest,
    config: LayoutConfiguration = ...,
    keep_knn: bool = ...,
    create_mst: bool = ...,
    clear_lsh_forest: bool = ...,
) -> tuple[VectorFloat, VectorFloat, VectorUint, VectorUint, GraphProperties]:
    """layout_from_lsh_forest(lsh_forest: tmap::LSHForest, config: _tmap.LayoutConfiguration = k: 10
    kc: 10
    fme_iterations: 1000
    fme_randomize: 0
    fme_threads: 4
    fme_precision: 4
    sl_repeats: 1
    sl_extra_scaling_steps: 2
    sl_scaling_x: 1.000000
    sl_scaling_y: 1.000000
    sl_scaling_type: RelativeToDrawing
    mmm_repeats: 1
    placer: Barycenter
    merger: LocalBiconnected
    merger_factor: 2.000000
    merger_adjustment: 0
    node_size0.015385, keep_knn: bool = False, create_mst: bool = True, clear_lsh_forest: bool = False) -> tuple[_tmap.VectorFloat, _tmap.VectorFloat, _tmap.VectorUint, _tmap.VectorUint, _tmap.GraphProperties].


            Create minimum spanning tree or k-nearest neighbor graph coordinates and topology from an :obj:`LSHForest` instance.

    Arguments:
                lsh_forest (:obj:`LSHForest`): An :obj:`LSHForest` instance

    Keyword Arguments:
                config (:obj:`LayoutConfiguration`, optional): An :obj:`LayoutConfiguration` instance
                create_mst (:obj:`bool`, optional): Whether to create a minimum spanning tree or to return coordinates and topology for the k-nearest neighbor graph
                clear_lsh_forest (:obj:`bool`, optional): Whether to run :obj:`clear()` on the :obj:`LSHForest` instance after k-nearest negihbor graph and MST creation and before layout

    Returns:
                :obj:`tuple[VectorFloat, VectorFloat, VectorUint, VectorUint, GraphProperties]` The x and y coordinates of the vertices, the ids of the vertices spanning the edges, and information on the graph

    """

def layout_from_lsh_forest_native(
    lsh_forest,
    config: LayoutConfiguration = ...,
    keep_knn: bool = ...,
    create_mst: bool = ...,
    clear_lsh_forest: bool = ...,
) -> tuple:
    """layout_from_lsh_forest_native(lsh_forest: tmap::LSHForest, config: _tmap.LayoutConfiguration = k: 10
    kc: 10
    fme_iterations: 1000
    fme_randomize: 0
    fme_threads: 4
    fme_precision: 4
    sl_repeats: 1
    sl_extra_scaling_steps: 2
    sl_scaling_x: 1.000000
    sl_scaling_y: 1.000000
    sl_scaling_type: RelativeToDrawing
    mmm_repeats: 1
    placer: Barycenter
    merger: LocalBiconnected
    merger_factor: 2.000000
    merger_adjustment: 0
    node_size0.015385, keep_knn: bool = False, create_mst: bool = True, clear_lsh_forest: bool = False) -> tuple.


            Create minimum spanning tree or k-nearest neighbor graph coordinates and topology from an :obj:`LSHForest` instance. This method returns native python lists and objects.

    Arguments:
                lsh_forest (:obj:`LSHForest`): An :obj:`LSHForest` instance

    Keyword Arguments:
                config (:obj:`LayoutConfiguration`, optional): An :obj:`LayoutConfiguration` instance
                create_mst (:obj:`bool`, optional): Whether to create a minimum spanning tree or to return coordinates and topology for the k-nearest neighbor graph
                clear_lsh_forest (:obj:`bool`, optional): Whether to run :obj:`clear()` on the :obj:`LSHForest` instance after k-nearest negihbor graph and MST creation and before layout

    Returns:
                :obj:`tuple[list, list, list, list, Object]` The x and y coordinates of the vertices, the ids of the vertices spanning the edges, and information on the graph

    """

def make_edge_list(
    x: VectorFloat, y: VectorFloat, s: VectorUint, t: VectorUint
) -> tuple[VectorFloat, VectorFloat, VectorFloat, VectorFloat]:
    """make_edge_list(x: _tmap.VectorFloat, y: _tmap.VectorFloat, s: _tmap.VectorUint, t: _tmap.VectorUint) -> tuple[_tmap.VectorFloat, _tmap.VectorFloat, _tmap.VectorFloat, _tmap.VectorFloat].

    Creates an edge list from x, y coordinates and edge indices.

    Arguments:
        x (:obj:`VectorFloat`): The x coordinates
        y (:obj:`VectorFloat`): The y coordinates
        s (:obj:`VectorUint`): The indices of the from vertices
        t (:obj:`VectorUint`): The indices of the to vertices

    Returns:
        :obj:`tuple[VectorFloat, VectorFloat, VectorFloat, VectorFloat]`: Coordinates in edge list form

    """

def make_edge_list_native(
    x: VectorFloat, y: VectorFloat, s: VectorUint, t: VectorUint
) -> tuple:
    """make_edge_list_native(x: _tmap.VectorFloat, y: _tmap.VectorFloat, s: _tmap.VectorUint, t: _tmap.VectorUint) -> tuple.

    brief Creates an edge list from x, y coordinates and edge indices. This method returns native python lists and objects. Also returns coordinates of vertices

    Arguments:
        x (:obj:`VectorFloat`): The x coordinates
        y (:obj:`VectorFloat`): The y coordinates
        s (:obj:`VectorUint`): The indices of the from vertices
        t (:obj:`VectorUint`): The indices of the to vertices

    Returns:
        :obj:`tuple[list, list, list, list, list, list]`: Coordinates in edge list form and the vertex coordinates

    """

def map(
    arr: np.ndarray[np.float64],
    dims: int = ...,
    n_trees: int = ...,
    dtype: str = ...,
    config: LayoutConfiguration = ...,
    file_backed: bool = ...,
    seed: int = ...,
) -> tuple:
    """map(arr: numpy.ndarray[numpy.float64], dims: int = 128, n_trees: int = 8, dtype: str = 'binary', config: _tmap.LayoutConfiguration = k: 10
    kc: 10
    fme_iterations: 1000
    fme_randomize: 0
    fme_threads: 4
    fme_precision: 4
    sl_repeats: 1
    sl_extra_scaling_steps: 2
    sl_scaling_x: 1.000000
    sl_scaling_y: 1.000000
    sl_scaling_type: RelativeToDrawing
    mmm_repeats: 1
    placer: Barycenter
    merger: LocalBiconnected
    merger_factor: 2.000000
    merger_adjustment: 0
    node_size0.015385, file_backed: bool = False, seed: int = 42) -> tuple.


            Create minimum spanning tree or k-nearest neighbor graph coordinates and topology from an :obj:`LSHForest` instance. This method returns native python lists and objects.

    Arguments:
                arr (:obj:`Array`): A numpy :obj:`Array` instance

    Keyword Arguments:
                dims (:obj:`int`, optional): The number of permutations to use for the MinHash algorithm
                n_trees (:obj:`int`, optional): The number of forests to use in the LSHForest data structure
                dtype (:obj:`str`, optional): The type of data that is supplied, can be 'binary', 'sparse', or 'weighted'
                config (:obj:`LayoutConfiguration`, optional): An :obj:`LayoutConfiguration` instance
                file_backed (:obj:`bool`) Whether to store the data on disk rather than in main memory (experimental)
                seed (:obj:`int`): The seed used for the random number generator(s)

    Returns:
                :obj:`tuple[list, list, list, list, Object]` The x and y coordinates of the vertices, the ids of the vertices spanning the edges, and information on the graph

    """

def mean_quality(gp: GraphProperties) -> VectorFloat:
    """mean_quality(gp: _tmap.GraphProperties) -> _tmap.VectorFloat.

    brief Calculates the mean quality of all vertices based on the actual nearest neighbors and the topological distances in the spanning tree.

    Arguments:
        gp (:obj:`VectorFloat`): A GraphProperties object

    Returns:
        :obj:`list[float]`: The average topological distances ordered by k-nearest neighbor.

    """

def mst_from_edge_list(
    vertex_count: int, edges: list[tuple[int, int, float]]
) -> tuple[VectorUint, VectorUint, VectorFloat]:
    """mst_from_edge_list(vertex_count: int, edges: list[tuple[int, int, float]]) -> tuple[_tmap.VectorUint, _tmap.VectorUint, _tmap.VectorFloat].

    Create minimum spanning tree topology from an edge list.

    Arguments:
        vertex_count (:obj:`int`): The number of vertices in the edge list
        edges (:obj:`list` of :obj:`tuple[int, int, float]`): An edge list defining a graph

    Returns:
        :obj:`tuple[VectorUint, VectorUint, VectorFloat]`: the topology of the minimum spanning tree of the data from the edge list

    """

def mst_from_lsh_forest(
    lsh_forest, k: int, kc: int = ...
) -> tuple[VectorUint, VectorUint, VectorFloat]:
    """mst_from_lsh_forest(lsh_forest: tmap::LSHForest, k: int, kc: int = 10) -> tuple[_tmap.VectorUint, _tmap.VectorUint, _tmap.VectorFloat].

    Create minimum spanning tree topology from an :obj:`LSHForest` instance.

    Arguments:
        lsh_forest (:obj:`LSHForest`): An :obj:`LSHForest` instance
        int k (:obj:`int`): The number of nearest neighbors used to create the k-nearest neighbor graph

    Keyword Arguments:
        int kc (:obj:`int`, optional): The scalar by which k is multiplied before querying the LSH forest. The results are then ordered decreasing based on linear-scan distances and the top k results returned

    Returns:
        :obj:`tuple[VectorUint, VectorUint, VectorFloat]`: the topology of the minimum spanning tree of the data indexed in the LSH forest

    """

def vertex_quality(gp: GraphProperties, v: int) -> list[tuple[int, float, int]]:
    """vertex_quality(gp: _tmap.GraphProperties, v: int) -> list[tuple[int, float, int]].

    brief Computes the visualization quality of a vertex based on it's true nearest neighbors and their distribution in the tree.

    Arguments:
        gp (:obj:`VectorFloat`): A GraphProperties object
        v (:obj:`int`): The vertex id of the node to be analysed

    Returns:
        :obj:`list[tuple[int, float, int]]`: The qualities based on the degrees in the knn graph.

    """
