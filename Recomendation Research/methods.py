from openai import OpenAI
from dotenv import load_dotenv
from typing import Callable
import matplotlib.pyplot as plt
from random import randint
import time
import numpy as np

import os
import re
import pickle

# Load the .env file
load_dotenv()
personal_api_key = os.environ.get('MY_API_KEY')

class Static:
    @staticmethod
    def message_for_city(city):
        return f"Recommend five hotels in {city} UK. \n State the name of the hotel on a new line each time not using a numbered list."
    
    @staticmethod
    def plot_from_ranks_target(pickle_path, target_words, filter: int = 3, key: str = None):
        lm = LMEO()

        with open(pickle_path, 'rb') as f:
            ranks = pickle.load(f)

        ranks = [rank for rank in ranks if len(rank) == 5] # Making sure everything has the correct rank.

        target_dict = lm.group_hotels(ranks, target_words)

        freq_dict = lm.frequency_in_row(ranks, target_dict, filter)

        if key is not None:
            hotel_id_list = list(freq_dict[key].keys())
        else:
            hotel_id_list = []
            for key in freq_dict:
                hotel_id_list.extend(freq_dict[key].keys())
            
            hotel_id_list = list(set(hotel_id_list))

        for hotel_id in hotel_id_list:
            hotel_id_key = hotel_id if type(hotel_id) == str else hotel_id[0]
            amount_in_place = [place.get(hotel_id_key, 0) for _, place in freq_dict.items()]
            
            if sum(amount_in_place) != 0:
                plt.plot([place.get(hotel_id_key, 0) for _, place in freq_dict.items()], label=hotel_id_key)
        
        if len(hotel_id_list) < 5:
            plt.legend()
        
        plt.show()

    @staticmethod
    def plot_from_ranks(ranks, target_words, filter: int = 3, key: str = None):
        lm = LMEO()

        ranks = [rank for rank in ranks if len(rank) == 5] # Making sure everything has the correct rank.

        target_dict = lm.group_hotels(ranks, target_words)

        freq_dict = lm.frequency_in_row(ranks, target_dict, filter)

        if key is not None:
            hotel_id_list = list(freq_dict[key].keys())
        else:
            hotel_id_list = []
            for key in freq_dict:
                hotel_id_list.extend(freq_dict[key].keys())
            
            hotel_id_list = list(set(hotel_id_list))

        for hotel_id in hotel_id_list:
            hotel_id_key = hotel_id if type(hotel_id) == str else hotel_id[0]
            amount_in_place = [place.get(hotel_id_key, 0) for _, place in freq_dict.items()]
            
            if sum(amount_in_place) != 0:
                plt.plot([place.get(hotel_id_key, 0) for _, place in freq_dict.items()], label=hotel_id_key)
        
        if len(hotel_id_list) < 5:
            plt.legend()
        
        plt.show()

class LMEO:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        personal_api_key = os.environ.get('MY_API_KEY')
        
        self.client = OpenAI(api_key=personal_api_key)
        self.model = model
  
    def order_set(self, message_content: str, message_identifier: str = None, times: int = 100, num_ranks=5, dump=True):
        ''' Generate multiple rank sets. '''

        if os.path.exists(fr'.\pickles\ranks_{message_identifier}.pkl'):
            print("Loading Previously Generated Ranks.")
            with open(fr'.\pickles\ranks_{message_identifier}.pkl', 'rb') as file:
                return pickle.load(file)
        
        message = [{"role": "user", "content": message_content}]
        
        try:
            choices_list = []
            chunk_size = 128
            full_chunks = times // chunk_size
            remainder = times % chunk_size

            for i in range(full_chunks):
                print(f'Ordering batch {i}.')
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    n=chunk_size
                )
                choices_list.extend(response.choices)

            if remainder > 0:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    n=remainder
                )
                choices_list.extend(response.choices)

            # Will remove most of the unwanted aretfacts from the response. By no means optimsed.
            pattern = r"\d+|\s?-\s|-\s|\s?\.\s|\s|,"
            ranks = [LLM_response.message.content.split("\n") for LLM_response in choices_list]

            # We should filter out here any ranks which are not of length length
            ranks = [rank for rank in ranks if len(rank) == num_ranks]

            ranks = [[re.sub(pattern, "", item) for item in rank] for rank in ranks]
            # remove any empty strings
            ranks = [[item for item in rank if item] for rank in ranks]
        
        except Exception as e:
            # We will dump the current choices_list in a breakout and then raise the exception:
            if len(choices_list) > 0:
                with open(fr'.\pickles\ranks_{message_identifier}_aborted.pkl', 'wb') as file:
                    pickle.dump(choices_list, file)
            raise e

        if dump:
            with open(fr'.\pickles\ranks_{message_identifier}.pkl', 'wb') as file:
                pickle.dump(ranks, file)
        
        return ranks
    
    def multiprompt_data(self, city_list, prompt_list, times):
        
        multi_prompt_data = {city: {} for city in city_list}
        multi_prompt_data["prompt_list"] = prompt_list
        multi_prompt_data["city_list"] = city_list

        # Collecting the data for each city.
        for city in city_list:
            for i, prompt in enumerate(prompt_list):
                formated_prompt = prompt.format(city)
                # Now generate the ranks
                rank_for_city_prompt = self.order_set(formated_prompt, times=times, dump=False)
                multi_prompt_data[city][i] = rank_for_city_prompt
        
        # Dumping the data
        with open(fr'.\pickles\multi_prompt_data_{randint(1,100)}.pkl', 'wb') as file:
            pickle.dump(multi_prompt_data, file)

    @staticmethod
    def group_hotels(ranks, target_words: list[str] = None, give_set_difference = False) -> dict[str]:
        ''' Group hotels by target words. '''

        unpacked_ranks = [hotel for rank in ranks for hotel in rank]
        all_hotels = set(unpacked_ranks)

        grouped_hotels = {}

        all_hotels_copy = all_hotels.copy()
        for word in target_words:
            if type(word) == list:
                # we will now check if any of the words are in the hotel name.
                relevant = [hotel for hotel in all_hotels if any(word in hotel for word in word)]
                grouped_hotels[word[0]] = relevant
            else:
                relevant = [hotel for hotel in all_hotels if word in hotel]
                grouped_hotels[word] = relevant
            
            all_hotels_copy = all_hotels_copy.difference(relevant)

        # All the hotels which werent grouped together.
        non_grouped = list(all_hotels_copy)
        
        for word in all_hotels_copy:
            grouped_hotels[word] = [word]

        if give_set_difference:
            return grouped_hotels, non_grouped
        else:
            return grouped_hotels
        
    @staticmethod
    def create_split_dict(all_ranks, target_words: list[str] = None, condition: Callable = lambda x: 1) -> dict[str]:

        # vectorised condition
        all_ranks = np.array(all_ranks)
        condition_mask = np.array([condition(rank) for rank in all_ranks])

        stacked_ranks_T, stacked_ranks_F = np.column_stack(all_ranks[condition_mask]), np.column_stack(all_ranks[~condition_mask])
        
        return LMEO.create_percent_dict(stacked_ranks_T, target_words), LMEO.create_percent_dict(stacked_ranks_F, target_words)

    @staticmethod
    def hotel_target_flag(hotel, target_word):
            if isinstance(target_word, list):
                return any(word in hotel for word in target_word)
            else:
                return target_word in hotel
    
    
    @staticmethod   
    def create_percent_dict(stacked_ranks, target_words, other = False):
            percent_dict = {i: {} for i in range(len(stacked_ranks))}

            # Itterating over the frequency lists in each place.
            for i, hotel_list in enumerate(stacked_ranks):
                num_hotels_i = len(hotel_list) # This is the total number of hotels and will be used for the percentage.
                
                for hotel_from_rank in hotel_list: # Not opimised but will do for now.
                    for target_word in target_words:
                        if LMEO.hotel_target_flag(hotel_from_rank, target_word):
                            id = target_word if type(target_word) == str else target_word[0]
                            prev_count = percent_dict[i].get(id, 0)
                            percent_dict[i][id] = prev_count + 1/num_hotels_i # turning into a percentage.
                            break
                    else:
                        # If the hotel is not in the target words
                        if other:
                            other_count = percent_dict[i].get("Other", 0)
                            percent_dict[i]['Other'] = other_count + 1/num_hotels_i
                        else:
                            other_count = percent_dict[i].get(hotel_from_rank, 0)
                            percent_dict[i][hotel_from_rank] = other_count + 1/num_hotels_i



            return percent_dict
    

    @staticmethod
    def frequency_in_row(ranks: list, grouped_hotels_dict: dict[str], filter_less: int =0,  num_ranks: int = 5):
        
        frequency_dict = {x: {} for x in range(num_ranks)}
        
        for rank_instance in ranks:
            for i, hotel in enumerate(rank_instance):
                for word, hotel_list in grouped_hotels_dict.items():
                    if hotel in hotel_list:
                        num_occurences = frequency_dict[i].get(word, 0)
                        frequency_dict[i][word] = num_occurences + 1

        # We will now filter out the hotels which have less than the filter_less occurences.
        for i in range(num_ranks):
            frequency_dict[i] = {key: value for key, value in frequency_dict[i].items() if value >= filter_less}
        
        return frequency_dict
    

