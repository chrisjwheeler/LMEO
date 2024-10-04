import numpy as np
import openai
import anthropic
import pickle
import itertools

from dotenv import load_dotenv

import os

load_dotenv()

def exp_transform(params):
    """Transform parameters into exp-scale weights."""
    weights = np.exp(np.asarray(params) - np.mean(params))
    return (len(weights) / weights.sum()) * weights

def log_transform(weights):
    """Transform weights into centered log-scale parameters."""
    params = np.log(weights)
    return params - params.mean()

import abc

class ConvergenceTest(metaclass=abc.ABCMeta):

    """Abstract base class for convergence tests.

    Convergence tests should implement a single function, `__call__`, which
    takes a parameter vector and returns a boolean indicating whether or not
    the convergence criterion is met.
    """

    @abc.abstractmethod
    def __call__(self, params, update=True):
        """Test whether convergence criterion is met.

        The parameter `update` controls whether `params` should replace the
        previous parameters (i.e., modify the state of the object).
        """

class NormOfDifferenceTest(ConvergenceTest):

    """Convergence test based on the norm of the difference vector.

    This convergence test computes the difference between two successive
    parameter vectors, and declares convergence when the norm of this
    difference vector (normalized by the number of items) is below `tol`.
    """

    def __init__(self, tol=1e-8, order=1):
        self._tol = tol
        self._ord = order
        self._prev_params = None

    def __call__(self, params, update=True):
        params = np.asarray(params) - np.mean(params)
        if self._prev_params is None:
            if update:
                self._prev_params = params
            return False
        dist = np.linalg.norm(self._prev_params - params, ord=self._ord)
        if update:
            self._prev_params = params
        return dist <= self._tol * len(params)


def _mm(n_items, data, initial_params, alpha, max_iter, tol, mm_fun):
    """
    Iteratively refine MM estimates until convergence.

    Raises
    ------
    RuntimeError
        If the algorithm does not converge after `max_iter` iterations.
    """
    
    if initial_params is None:
        params = np.zeros(n_items)
    else:
        params = initial_params
    converged = NormOfDifferenceTest(tol=tol, order=1)
    
    for _ in range(max_iter):
        nums, denoms = mm_fun(n_items, data, params)
        params = log_transform((nums + alpha) / (denoms + alpha))
        if converged(params):
            return params
        
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))

def _mm_pairwise(n_items, data, params):
    """Inner loop of MM algorithm for pairwise data."""
    weights = exp_transform(params)
    wins = np.zeros(n_items, dtype=float)
    denoms = np.zeros(n_items, dtype=float)
    for winner, loser in data:
        wins[winner] += 1.0
        val = 1.0 / (weights[winner] + weights[loser])
        denoms[winner] += val
        denoms[loser] += val
    return wins, denoms


def mm_pairwise(
        n_items, data, initial_params=None, alpha=0.0,
        max_iter=10000, tol=1e-8):
    """Compute the ML estimate of model parameters using the MM algorithm.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given pairwise-comparison data (see :ref:`data-pairwise`), using
    the minorization-maximization (MM) algorithm [Hun04]_, [CD12]_.

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under a (peaked) Dirichlet prior. See :ref:`regularization` for
    details.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Pairwise-comparison data.
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
    alpha : float, optional
        Regularization parameter.
    max_iter : int, optional
        Maximum number of iterations allowed.
    tol : float, optional
        Maximum L1-norm of the difference between successive iterates to
        declare convergence.

    Returns
    -------
    params : numpy.ndarray
        The ML estimate of model parameters.
    """
    return _mm(
            n_items, data, initial_params, alpha, max_iter, tol, _mm_pairwise)


def _mm_rankings(n_items, data, params):
    """Inner loop of MM algorithm for ranking data."""
    weights = exp_transform(params)
    wins = np.zeros(n_items, dtype=float)
    denoms = np.zeros(n_items, dtype=float)
    for ranking in data:
        sum_ = weights.take(ranking).sum()
        for i, winner in enumerate(ranking[:-1]):
            wins[winner] += 1
            val = 1.0 / sum_
            for item in ranking[i:]:
                denoms[item] += val
            sum_ -= weights[winner]
    return wins, denoms


def mm_rankings(n_items, data, initial_params=None, alpha=0.0,
            max_iter=10000, tol=1e-8):
        """Compute the ML estimate of model parameters using the MM algorithm.

        This function computes the maximum-likelihood (ML) estimate of model
        parameters given ranking data (see :ref:`data-rankings`), using the
        minorization-maximization (MM) algorithm [Hun04]_, [CD12]_.

        If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
        estimate under a (peaked) Dirichlet prior. See :ref:`regularization` for
        details.

        Parameters
        ----------
        n_items : int
            Number of distinct items.
        data : list of lists
            Ranking data.
        initial_params : array_like, optional
            Parameters used to initialize the iterative procedure.
        alpha : float, optional
            Regularization parameter.
        max_iter : int, optional
            Maximum number of iterations allowed.
        tol : float, optional
            Maximum L1-norm of the difference between successive iterates to
            declare convergence.

        Returns
        -------
        params : numpy.ndarray
            The ML estimate of model parameters.
        """
        return _mm(n_items, data, initial_params, alpha, max_iter, tol,
                _mm_rankings)

def _mm_top1(n_items, data, params):
    """Inner loop of MM algorithm for top1 data."""
    weights = exp_transform(params)
    wins = np.zeros(n_items, dtype=float)
    denoms = np.zeros(n_items, dtype=float)
    for winner, losers in data:
        wins[winner] += 1
        val = 1 / (weights.take(losers).sum() + weights[winner])
        for item in itertools.chain([winner], losers):
            denoms[item] += val
    return wins, denoms


def mm_top1(
        n_items, data, initial_params=None, alpha=0.0,
        max_iter=10000, tol=1e-8):
    """Compute the ML estimate of model parameters using the MM algorithm.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given top-1 data (see :ref:`data-top1`), using the
    minorization-maximization (MM) algorithm [Hun04]_, [CD12]_.

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under a (peaked) Dirichlet prior. See :ref:`regularization` for
    details.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Top-1 data.
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
    alpha : float, optional
        Regularization parameter.
    max_iter : int, optional
        Maximum number of iterations allowed.
    tol : float, optional
        Maximum L1-norm of the difference between successive iterates to
        declare convergence.

    Returns
    -------
    params : numpy.ndarray
        The ML estimate of model parameters.
    """
    return _mm(n_items, data, initial_params, alpha, max_iter, tol, _mm_top1)

class SimilarityTests:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        personal_api_key = os.environ.get('MY_API_KEY')
        anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        self.client = openai.OpenAI(api_key=personal_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.model = model

    @staticmethod
    def get_prompt_dict():
        ''' Returns the default prompt for the similarity tests.'''
        return {'pairwise': 'Which hotel do you recommend more: {} or {}. Simply write the name of the hotel and nothing else.',
                'topk': 'Recommend {} hotels in {}. State the name of the hotel and nothing else.',
                'top1': 'Recommend one hotel in {}. State the name of the hotel and nothing else.'}
    
    @staticmethod
    def get_words_dict() -> dict:
        '''Returns a dict with target_words for each city.'''

        words_dict = {
            'Exeter': [['Abode', 'ABode']] + 'Vin Mercure Holiday Queens Gate Devon Magdalen South Globe Rougemont Buckerel'.split(),
            'Bristol': ['Marriott','Gainsborough','Hilton','Mercure', 'arbour' ,'Radisson', 'Avon', 'DoubleTree', 'Berkeley', 'ibis', 'Future','ztec', 'gabel'],
            'London': 'Savoy Langham Ritz Ned Clarid hangri Dorchest Shard Rose Royal'.split(),
            'Manchester': 'Gotham rincipal idland adisson ilton'.split() + ['Meli', 'inn', 'INN', 'Inn'],
            'Newcastle': 'Crowne Grey Malmaison Dene Vin County Dene Sandman Indigo Hilton Jury'.split(),
            'Brighton': 'Drake Vin Jurys Artist Queens Grand TheBrighton'.split(),
            'Birmingham': 'ACHotel Hyatt Metro Marriot adisson almaison BirminghamCityCentre Vin Cube dgbaston Grand Staying Indigo Hampton'.split(),
            'Leeds': 'Dakota DoubleTree Malmaison Marrio Quebecs Radisson Queen Thorpe Park bisSt Cosmo'.split(),
            'Liverpool': 'Double HopeS James Pullman Liner Titanic Indigo Richmond Malmaison'.split()
        }

        return words_dict

    @staticmethod
    def hotel_target_flag(hotel, target_word):
            if isinstance(target_word, list):
                return any(word in hotel for word in target_word)
            else:
                return target_word in hotel
            
    @staticmethod
    def get_conversions(target_words, other_flag=True):
        ''' Returns the conversion dicts for the target words.'''
        inner_target_words = target_words + ['OTHER'] if other_flag else target_words

        # Coversion dicts.
        num_to_id = {i : x[0] if isinstance(x, list) else x for i, x in enumerate(inner_target_words)}
        id_to_num = {v : k for k, v in num_to_id.items()}

        return num_to_id, id_to_num
            

    def rank_to_num(self, ranks, target_words, other_flag=True):
        ''' Convert the ranks to a number based on a target word list possibly containing lists.
        target_words: Should be chosen so that other is used as comonly on average as each target word.'''
        
        # Adding category: OTHER to the target words.
        if other_flag:
            inner_target_words = target_words + ['OTHER']
        else:
            inner_target_words = target_words

        # Coversion dicts.
        num_to_id = {i : x[0] if isinstance(x, list) else x for i, x in enumerate(inner_target_words)}
        id_to_num = {v : k for k, v in num_to_id.items()}

        # Loop through each rank and covert to number.
        converted_ranks = []
        for rank_instance in ranks:
            converted_rank_instance = []
            
            for hotel in rank_instance:
                for target_id in inner_target_words:
                    # Will be used to convert the target word to a number.
                    single_id = target_id[0] if isinstance(target_id, list) else target_id
                    
                    if self.hotel_target_flag(hotel, target_id):
                        converted_rank_instance.append(id_to_num[single_id])
                        break
                else: # The hotel is in OTHER category.
                    converted_rank_instance.append(id_to_num['OTHER'])
            
            converted_ranks.append(converted_rank_instance)

        return converted_ranks, num_to_id
    
    def order_pairwise_preferences(self, pickle_id, pairwise_prompt, proper_hotel_ids, num_time=50, dump=True) -> dict:
        ''' Function to order the pairwise preferences off of the LLM. 
        
        Parameters:
        pairwise_prompt: str
            The prompt to ask the user to compare two hotels.
        hotel_name_id_tuple: This is the proper name of the hotel zipped with its id'''

        if os.path.exists(fr'.\pickles\pairwise_{pickle_id}.pkl'):
            print("Loading Previously Generated Ranks.")
            with open(fr'.\pickles\pairwise_{pickle_id}.pkl', 'rb') as file:
                return pickle.load(file)

        # We want all the permutation of the hotels where they are not (i,i)
        hotel_order_permutations = [(hotel_1, hotel_2) for hotel_1 in proper_hotel_ids for hotel_2 in proper_hotel_ids if hotel_1 != hotel_2]
        
        all_responses = []
        undecided = []

        for hotel_1, hotel_2 in hotel_order_permutations:
            # Create the pairwise payload.
            coresponding_pairwise_prompt = pairwise_prompt.format(hotel_1, hotel_2)
            LLM_payload = [{"role": "user", "content": coresponding_pairwise_prompt}]

            # Getting the response from the payload
            response = self.client.chat.completions.create(
                    model=self.model,
                    messages=LLM_payload,
                    n=num_time,
                )
            
            # Add the responses to the response_dict
            for message_obj in response.choices:
                preference = message_obj.message.content    

                if hotel_1 in preference:
                    all_responses.append([hotel_1, hotel_2])
                
                elif hotel_2 in preference:
                    all_responses.append([hotel_2, hotel_1])
                
                else:
                    undecided.append(preference)

        if dump:
            with open(fr'.\pickles\pairwise_{pickle_id}.pkl', 'wb') as file:
                pickle.dump(all_responses, file)
        
        print(f'Undecided: {undecided}')
        return all_responses  

    def order_top_1(self, pickle_id, prompt, times = 100, dump=True) -> list:

        if os.path.exists(fr'.\pickles\top1_{pickle_id}.pkl'):
            print("Loading Previously Generated Ranks.")
            with open(fr'.\pickles\top1_{pickle_id}.pkl', 'rb') as file:
                return pickle.load(file)

        LLM_payload = [{"role": "user", "content": prompt}]

        # Functionality to deal with requests which are greater than the chunk size.

        choices_list = []
        chunk_size = 128
        full_chunks = times // chunk_size
        remainder = times % chunk_size

        for i in range(full_chunks):
            print(f'Ordering batch {i}.')
            response = self.client.chat.completions.create(
                model=self.model,
                messages=LLM_payload,
                n=chunk_size
            )
            choices_list.extend(response.choices)

        if remainder > 0:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=LLM_payload,
                n=remainder
            )
            choices_list.extend(response.choices)

        all_responses = [[message_obj.message.content] for message_obj in choices_list]

        if dump:
            with open(fr'.\pickles\top1_{pickle_id}.pkl', 'wb') as file:
                pickle.dump(all_responses, file)

        return all_responses  
    
    def top1_ranks_to_params(self, ranks, target_words, **kargs):
        ''' Takes a sets of pairwise ranks and converts them to MM parameters.'''
        converted_ranks, num_to_id = self.rank_to_num(ranks, target_words, other_flag=True)
        # We will create a list of the form [rank, index_set] for each rank.
        
        formated_ranks = []
        for rank in converted_ranks:
            new_index_set = [i for i in range(len(num_to_id)) if i != rank[0]]

            formated_ranks.append([rank[0], new_index_set])

        MM_fitted_params = mm_top1(len(num_to_id), formated_ranks, **kargs) 
        MM_fitted_params = sorted(zip(MM_fitted_params, num_to_id.values()), key=lambda x: x[0], reverse=True)
        
        return MM_fitted_params
    
    def topk_ranks_to_params(self, ranks, target_words, **kargs):
        ''' Takes a sets of Topk ranks and converts them to MM parameters.'''
        converted_ranks, num_to_id = self.rank_to_num(ranks, target_words)
        
        MM_fitted_params = mm_rankings(len(num_to_id), converted_ranks, **kargs)
        MM_fitted_params = sorted(zip(MM_fitted_params, num_to_id.values()), key=lambda x: x[0], reverse=True)
        
        return MM_fitted_params
    
    def pairwise_ranks_to_params(self, ranks, target_words, **kargs):
        ''' Takes a sets of pairwise ranks and converts them to MM parameters.'''
        converted_ranks, num_to_id = self.rank_to_num(ranks, target_words, other_flag=False)
        
        MM_fitted_params = mm_pairwise(len(num_to_id), converted_ranks, **kargs) 
        MM_fitted_params = sorted(zip(MM_fitted_params, num_to_id.values()), key=lambda x: x[0], reverse=True)
        
        return MM_fitted_params
    