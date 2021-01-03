###############################################################################################################################################
##Environment classes
###############################################################################################################################################
from enum import Enum

class Actions(Enum):
    HOLD = 0
    BUY = 1
    SELL = -1

class MarketEnv():
    def __init__(self, df, window_size, capital, lot_size):
        # Set attributes
        self.df = df                        # Dask dataframe
        self.window_size = window_size
        self.initial_capital = capital
        self.action_space_size = len(Actions)
        #self.leverage = leverage

        self.done = None
        self._index = None
        self.df_length = len(df)
        self.position = None
        self.profit = None
        self.price = None
        self.current_state = None

        # Check dataframe
        self._check_df()

    def reset(self):
        self._index = 0
        self.done = False
        self.capital = self.initial_capital
        self.position = 0 # Number of lot sizes - positive = long, negative = short
        self.profit = 0
        
        self.price, self.current_state = self._get_states()

        return self.current_state

    def _check_df(self):
        pass

    def _get_states(self):
        return None, None

class SharesMarketEnv(MarketEnv):
    def _check_df(self):
        # Check if dataframe is the expected format
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if not all(expected_columns == self.df.columns):
            raise Exception('Dataframe does not have expected column format')

    def _get_states(self):
        # Extract window for observation and price (adjusted close)
        window = self.df[self.df.columns].values[list(range(self._index, self.window_size))].compute()
        price = [window[-1][4]]

        return price, window

    # Simple buy and sell entire capital implementation - assumes that orders made at market open after latest observation
    def step(self, action):
        # Update position
        if (position == 0 and action != 0):
            position = action

        # 
        



class ForexMarketEnv(MarketEnv):
    def _check_df(self):
        # Check if dataframe is the expected format
        expected_columns = ['B_Open','B_High','B_Low','B_Close','A_Open','A_High','A_Low','A_Close']
        if not all(expected_columns == self.df.columns):
            raise Exception('Dataframe does not have expected column format')

    '''
    Current implementation is aggregation mode with constant lot size (specified when object iniitlised)
    Orders are executed at the exact price shown to the agent on the previous step ----- NEEDS MORE ACCURATE IMPLEMENTATION
    Profits/Losses calculated from difference in price from previous timestep to current
    Account currency not taken into account - no messy currency changes etc.
    '''
    def step(self, action):
        raise Exception('Forex market environment not yet implemented')

    def _get_states(self):
        # Extract window for observation and price ([Bid, Ask])
        window = self.df[self.df.columns].values[list(range(self._index, self.window_size))].compute()
        price = [window[-1][3], window[-1][7]]

        return price, window


    