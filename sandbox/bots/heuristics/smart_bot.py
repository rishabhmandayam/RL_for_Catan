from catanatron.players.value import ValueFunctionPlayer, CONTENDER_WEIGHTS

class SmartBot(ValueFunctionPlayer):
    """
    A customized Heuristic Player.
    Inherits from ValueFunctionPlayer which uses a set of weights to evaluate
    the game state and pick the action leading to the highest value.
    
    You can tweak CONTENDER_WEIGHTS or define your own base_fn.
    """
    def __init__(self, color, is_bot=True, epsilon=0.0):
        # 'C' loads the contender_fn which uses CONTENDER_WEIGHTS
        super().__init__(color, value_fn_builder_name="C", params=CONTENDER_WEIGHTS, is_bot=is_bot, epsilon=epsilon)

    def __str__(self):
        return f"SmartBot({self.color})"
