
class GetVariableValues:

    @staticmethod
    def get_nums(var_symbol: str) -> list[str]:
        '''
        input: var_symbol = the symbol of the variable (e.g. Y, V_1, etc etc.)
        output: a list of the numerical values (e.g. 1, 2, 3...) for that variable
        ''' 
        vv_dic = VariableValues()
        var = vv_dic.Variables_dic[var_symbol]
        vals_list = []
        for v in list(var.keys()):
            vals_list.append(str(v))
        return vals_list
            
    @staticmethod
    def get_labels(var_symbol: str) -> list[str]:
        '''
        input: var_symbol = the symbol of the variable (e.g. Y, V_1 etc.)
        output: a list of the label values (e.g. "End terrace", "Mid terrace"...) for that variable
        '''
        vv_dic = VariableValues()
        var = vv_dic.Variables_dic[var_symbol]
        vals_list = []
        for v in list(var.values()):
            vals_list.append(str(v))
        return vals_list       



class VariableValues():
    '''
    Variables dictionaries: the values for each observed variable are represened as key-value pairs in a python dictionary.
    - The dict-key corresponds to the "number" of the Variable's value (which is entered in the csv datased).
    - The dict-value correspond to the "label" of the Variable's value. 
    '''
    Variables_dic: dict = None


    def __init__(self):  
        '''
        Discrete variables (categorical, ordinal, etc.)
        ''' 
        # Household income
        V_1_values = {
            1: "0-6000",
            2: "6001-10000",
            3: "10001-15000",
            4: "15001-20000",
            5: "20001-25000",
            6: "25001-30000",
            7: "30001-40000",
            8: ">4000"
        }

        # Fleet size
        V_2_values = {
            0: "no car",
            1: "1 car",
            2: "2 cars or more"
        }

        # No of adults in household
        V_3_values = {
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5"
        }

        # No of childred in household
        V_4_values = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4"
        }

        # Household composition
        V_5_values = {
            1: "Single adult",
            2: "Small multiple adults",
            3: "Single parent",
            4: "Small family",
            5: "Large family",
            6: "Large multiple adults",
            7: "Small old adults",
            8: "Single pensioner"
        }

        # household urban/rural classification
        V_6_values = {
            1: "urban",
            2: "rural"
        }
        
        # parking provision
        V_7_values = {
            1: "Off-street parking",
            2: "On-street parking"
        }
        
        # Type of dwelling
        V_8_values = {
            1: "detached house",
            2: "semi-detached house",
            3: "terrace house",
            4: "tenement flat",
            5: "4-in-a-block flat",
            6: "tower/slab flat",
            7: "flat from converted house"
        }
        
        # Dwelling age
        V_9_values = {
            1: "Pre 1919",
            2: "1919-1944",
            3: "1945-1964",
            4: "1965-1982",
            5: "1983-2002",
            6: "post-2002"
        }
        
        # Local authority
        V_10_values = {
            1: "South Ayrshire",
            2: "South Lanarkshire",
            3: "Stirling",
            4: "West Dumbartonshire",
            5: "West Lothian",
            6: "Na h-Eileanan Siar",
            'A': "Aberdeen City",
            'B': "Aberdeenshire",
            'C': "Angus",
            'D': "Argyll and Bute",
            'E': "Scottish Borders",
            'F': "Clackmannanshire",
            'G': "Dumfries and Galloway",
            'H': "Dundee City",
            'I': "East Ayrshire",
            'J': "East Dumbartonshire",
            'K': "East Lothian",
            'L': "East Renfrewshire",
            'M': "City of Edinburgh",
            'N': "Falkirk",
            'O': "Fife",
            'P': "Glasgow City",
            'Q': "Highland",
            'R': "Inverclyde",
            'S': "Midlothian",
            'T': "Moray",
            'U': "North Ayrshire",
            'V': "North Lanarkshire",
            'W': "Orkney Islands",
            'X': "Perth and Kinross",
            'Y': "Renfrewshire",
            'Z': "Shetland Islands"
        }
        
        
        # Whether random adult in household considers buying (or already has) a plug-in electric car or van
        Y_values = {
            1: "Already own electric car/van",
            2: "Thinking to buy one soon",
            3: "Thinking to buy one in the future",
            4: "Not considering to buy one",
            5: "Do not drive/need a car"
        }
        
        
        '''
        Dictionary of all observed variables where: 
        - dict-key = variable symbol
        - dict-value = variable dictionary
        '''
        Variables_dic = {
            'Y': Y_values,
            'V_1': V_1_values,
            'V_2': V_2_values,
            'V_3': V_3_values,
            'V_4': V_4_values,
            'V_5': V_5_values,
            'V_6': V_6_values,
            'V_7': V_7_values,
            'V_8': V_8_values,
            'V_9': V_9_values,
            'V_10': V_10_values,
        }

        self.Variables_dic = Variables_dic

        return

