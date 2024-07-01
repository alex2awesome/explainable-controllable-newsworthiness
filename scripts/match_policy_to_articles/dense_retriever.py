from math import ceil
from tqdm.auto import tqdm
import torch
from torch.nn.functional import normalize
import faiss
from retriv.paths import index_path, docs_path, dr_state_path, embeddings_folder_path
from retriv.dense_retriever.dense_retriever import DenseRetriever
from retriv.dense_retriever.encoder import Encoder, pbar_kwargs
from retriv.dense_retriever.ann_searcher import ANN_Searcher
from typing import List, Iterable, Union, Callable
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch import Tensor
import logging


from urllib.parse import urlparse, urljoin
import logging

logger = logging.getLogger(__name__)


def make_inverse_index_url(id_mapping):
    """
    Constructs an inverse index for the id_mapping, where the keys are urls + their cleaned versions
    and the values are the ids.

    """
    def clean_url(to_get_url):
        return urljoin(to_get_url, urlparse(to_get_url).path)

    inverse_index = {}
    for id, url in id_mapping.items():
        inverse_index[url] = id
        cleaned_url = clean_url(url)
        if cleaned_url not in inverse_index:
            inverse_index[cleaned_url] = id
        if cleaned_url[-1] == '/':
            cleaned_url = cleaned_url[:-1]
            if cleaned_url not in inverse_index:
                inverse_index[cleaned_url] = id

    return inverse_index


def get_search_params(subset_ids, index_info):
    """
    Helper function to configure and return the search parameters for a FAISS index based on provided index information.

    This function dynamically selects the appropriate FAISS search parameters class
    based on the type of index key indicated in `index_info`. It also parses additional
    parameters specified as a string and sets them appropriately.

    Parameters:
    subset_ids (list[int]): A list of subset IDs for which search parameters are to be configured.
    index_info (dict): Dictionary containing the index key and additional parameter string.
                       Expected keys are 'index_key' and 'index_param'.

    Returns:
    object: An instance of a FAISS search parameters class configured with the provided selector
            and additional parameters.
    """
    # Extract the index key from the index information dictionary.
    index_key = index_info["index_key"]

    # Choose the appropriate search parameters class based on the index key.
    params_class = (
        faiss.SearchParametersIVF if "IVF" in index_key else
        faiss.SearchParametersPQ if "PQ" in index_key else
        faiss.SearchParametersHNSW if 'HNSW' in index_key else
        faiss.SearchParameters
    )
    # Retrieve additional index parameters from the index information dictionary.
    index_kwargs = index_info['index_param']
    # Initialize an empty dictionary to store additional parameters.
    add_params = {}
    # Split the index parameters string by commas and iterate over each parameter.
    for p in index_kwargs.split(','):
        # Split each parameter into key and value pairs on the '=' character.
        k, v = p.split('=')
        # If the value is a digit, convert it from string to integer.
        if v.isdigit():
            v = int(v)
        add_params[k] = v

    # Create a selector for a batch of IDs using the provided subset_ids.
    sel = faiss.IDSelectorBatch(subset_ids)
    return params_class(sel=sel, **add_params)


def last_token_pool(
        last_hidden_states: Tensor,
        attention_mask: Tensor
) -> Tensor:
    """
    Helper method for Salesforce/SFR-Embedding-2_R

    Parameters
    ----------
    last_hidden_states
    attention_mask

    Returns
    -------

    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


pooling_methods = {
    "nvidia/NV-Embed-v1": "model_specific_encoder",
    "Salesforce/SFR-Embedding-2_R": "last_token_pool",
}

class MyEncoder(Encoder):
    def __init__(
            self,
            index_name: str = "new-index",
            model: str = "sentence-transformers/all-MiniLM-L6-v2",
            normalize: bool = True,
            return_numpy: bool = True,
            max_length: int = None,
            device: str = "cpu",
            hidden_size: int = None,
            pooling_method: str = None,
    ):

        logging.info(f"Initializing MyEncoder with model: {model}")
        ind_path = index_path(index_name)
        logging.info(f"Collections Path: {ind_path}")

        self.index_name = index_name
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(model, trust_remote_code=True).to(device).eval()
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)

        # Set the hidden size based on the model configuration if not provided.
        if hasattr(config, 'hidden_size') and hidden_size is None:
            self.embedding_dim = config.hidden_size
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size') and hidden_size is None:
            self.embedding_dim = config.text_config.hidden_size
        else:
            self.embedding_dim = hidden_size

        # Set the maximum length based on the model configuration if not provided.
        if hasattr(config, 'max_position_embeddings') and max_length is None:
            self.max_length = config.max_position_embeddings
        elif (
                hasattr(config, 'text_config') and
                hasattr(config.text_config, 'max_position_embeddings') and
                max_length is None
        ):
            self.max_length = config.text_config.max_position_embeddings
        else:
            self.max_length = max_length

        # Set the pooling method based on the model type if not provided.
        if pooling_method is None:
            self.pooling_method = pooling_methods.get(model, "mean_pooling")
        else:
            self.pooling_method = pooling_method

        self.normalize = normalize
        self.return_numpy = return_numpy
        self.device = device
        self.tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }

    def embed(self, texts_to_embed: List):
        with torch.no_grad():
            # Encode the texts using the specified pooling method.
            if self.pooling_method == "mean_pooling":
                tokens = self.tokenize(texts_to_embed)
                emb = self.encoder(**tokens).last_hidden_state
                emb = self.mean_pooling(emb, tokens["attention_mask"])
            elif self.pooling_method == "model_specific_encoder":
                emb = self.encoder.encode(texts_to_embed)
            elif self.pooling_method == "last_token_pool":
                batch_dict = self.tokenizer(
                    texts_to_embed, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                outputs = self.encoder(**batch_dict)
                emb = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # Normalize the embeddings if specified.
            if self.normalize:
                emb = normalize(emb, dim=-1)

        return emb

    def bencode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ):
        embeddings = []
        pbar = tqdm(
            total=len(texts),
            desc="Generating embeddings",
            disable=not show_progress,
            **pbar_kwargs,
        )
        for i in range(ceil(len(texts) / batch_size)):
            start, stop = i * batch_size, (i + 1) * batch_size
            emb = self.embed(texts[start:stop])
            embeddings.append(emb)
            pbar.update(stop - start)
        pbar.close()

        embeddings = torch.cat(embeddings)
        if self.return_numpy:
            embeddings = embeddings.detach().cpu().numpy()
        return embeddings


class MyDenseRetriever(DenseRetriever):
    """Narrow class to subset ids from `DenseRetriever`.

    The MAIN PURPOSE is to extend DenseRetriever so you can filter down the returned results by a list of ids.

    This class is specifically designed to handle `faiss` classes indexed with HNSW.
    It can be extended for other types with appropriate type-checking. The class allows
    for creation of inverse indices and implements functionalities to index and search
    collections efficiently using an ANN (Approximate Nearest Neighbors) approach.
    """
    def __init__(
            self,
            index_name: str = "new-index",
            model: str = "sentence-transformers/all-MiniLM-L6-v2",
            normalize: bool = True,
            max_length: int = None,
            embedding_dim: int = None,
            use_ann: bool = True,
            make_inverse_index: Union[None, Callable] = None,
            device: str = "cpu",
            *args,
            **kwargs
    ):
        """Initialize MyDenseRetriever with the option to create an inverse index.

        Args:
            index_name (str, optional): [retriv](https://github.com/AmenRa/retriv) will use `index_name` as the identifier of your index. Defaults to "new-index".
            model (str, optional): defines the encoder model to encode queries and documents into vectors. You can use an [HuggingFace's Transformers](https://huggingface.co/models) pre-trained model by providing its ID or load a local model by providing its path.  In the case of local models, the path must point to the directory containing the data saved with the [`PreTrainedModel.save_pretrained`](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) method. Note that the representations are computed with `mean pooling` over the `last_hidden_state`. Defaults to "sentence-transformers/all-MiniLM-L6-v2".
            normalize (bool, optional): whether to L2 normalize the vector representations. Defaults to True.
            max_length (int, optional): texts longer than `max_length` will be automatically truncated. Choose this parameter based on how the employed model was trained or is generally used. Defaults to 128.
            use_ann (bool, optional): whether to use approximate nearest neighbors search. Set it to `False` to use nearest neighbors search without approximation. If you have less than 20k documents in your collection, you probably want to disable approximation. Defaults to True.
            make_inverse_index: a callable that takes a dictionary and returns its inverse. Defaults to None.
        """
        self.index_name = index_name
        self.model = model
        self.normalize = normalize
        self.use_ann = use_ann
        self.device = device

        self.encoder = MyEncoder(
            index_name=index_name,
            model=model,
            normalize=normalize,
            max_length=max_length,
            hidden_size=embedding_dim,
            device=device,
        )

        if max_length is None:
            self.max_length = self.encoder.max_length
        else:
            self.max_length = max_length

        self.ann_searcher = ANN_Searcher(index_name=index_name)
        self.id_mapping = None
        self.doc_count = None
        self.doc_index = None
        self.embeddings = None
        self.id_mapping_reverse = None
        if make_inverse_index is None:
            self.make_inverse_index = lambda x: {v: k for k, v in x.items()}
        else:
            self.make_inverse_index = kwargs['make_inverse_index']

    def index(
        self,
        collection: Iterable,
        callback: callable = None,
        show_progress: bool = True,
        batch_size: int = 1,
    ):
        """Indexes the provided collection and generates an inverse index mapping.

        Args:
            collection (Iterable): The collection of documents to index.
            callback (callable, optional): A callback function for progress updates. Defaults to None.
            show_progress (bool, optional): Flag to show progress of indexing. Defaults to True.
        """
        if self.device == 'cuda':
            use_gpu = True
        else:
            use_gpu = False

        super().index(
            collection=collection,
            callback=callback,
            show_progress=show_progress,
            batch_size=batch_size,
            use_gpu=use_gpu
        )
        self.id_mapping_reverse = self.make_inverse_index(self.id_mapping)

    def search(
            self,
            query: str,
            include_id_list: List[str]=None,
            return_docs: bool = True,
            cutoff: int = 100,
            verbose: bool = False,
    ):
        """Searches the indexed collection using the given query.

        Args:
            query (str): The query string to search for.
            include_id_list (List[str], optional): List of ids to include in the search. Defaults to None.
            return_docs (bool, optional): Whether to return full documents or just ids and scores. Defaults to True.
            cutoff (int, optional): The number of results to return. Defaults to 100.
            verbose (bool, optional): If set to True, outputs additional log messages. Defaults to False.

        Returns:
            Either a list of documents or a dictionary of ids and their corresponding scores, based on return_docs.
        """
        encoded_query = self.encoder(query)
        encoded_query = encoded_query.reshape(1, len(encoded_query))

        if self.use_ann:
            if include_id_list is not None:
                internal_subset_ids = []
                for reverse_id in include_id_list:
                    if reverse_id in self.id_mapping_reverse:
                        internal_subset_ids.append(self.id_mapping_reverse[reverse_id])
                    else:
                        if verbose:
                            logger.warning(f'Warning: {reverse_id} not in id_mapping')
                search_params = get_search_params(internal_subset_ids, self.ann_searcher.faiss_index_infos)
            else:
                search_params = None
            scores, doc_ids = self.ann_searcher.faiss_index.search(encoded_query, cutoff, params=search_params)
            doc_ids, scores = doc_ids[0], scores[0]
            to_keep = list(filter(lambda x: x[0] != -1, zip(doc_ids, scores)))
            if len(to_keep) > 0:
                doc_ids, scores = zip(*to_keep)
            else:
                doc_ids, scores = [], []
        else:
            raise NotImplementedError('use ANN....')

        doc_ids = self.map_internal_ids_to_original_ids(doc_ids)
        return (
            self.prepare_results(doc_ids, scores)
            if return_docs
            else dict(zip(doc_ids, scores))
        )

    @staticmethod
    def load(
            index_name: str = "new-index",
            make_inverse_index: Union[None, Callable] = None,
            *args,
            **kwargs
    ):
        """Static method to load a previously saved index and its associated retriever.

        Args:
            index_name (str, optional): Name of the index to load. Defaults to "new-index".
            make_inverse_index (Union[None, Callable], optional): Function to generate an inverse index.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            MyDenseRetriever: The loaded dense retriever object.
        """
        state = np.load(dr_state_path(index_name), allow_pickle=True)["state"][()]
        dr = MyDenseRetriever(**state["init_args"])
        dr.initialize_doc_index()
        dr.id_mapping = state["id_mapping"]
        dr.doc_count = state["doc_count"]
        if state["embeddings"]:
            dr.load_embeddings()
        if dr.use_ann:
            dr.ann_searcher = ANN_Searcher.load(index_name)

        if 'id_mapping_reverse' not in state:
            if make_inverse_index is None:
                dr.id_mapping_reverse = {v: k for k, v in dr.id_mapping.items()}
            else:
                dr.id_mapping_reverse = make_inverse_index(dr.id_mapping)
        else:
            dr.id_mapping_reverse = state['id_mapping_reverse']

        return dr
