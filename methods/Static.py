from .LMEO import LMEO

import matplotlib.pyplot as plt
import pickle

class Static:
    @staticmethod
    def message_for_city(city):
        return f"Recommend five hotels in {city} UK. \n State the name of the hotel on a new line each time not using a numbered list."
    
    @staticmethod
    def get_words_dict() -> dict:
        '''Simply returns a dict with the most uptodata words.'''

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