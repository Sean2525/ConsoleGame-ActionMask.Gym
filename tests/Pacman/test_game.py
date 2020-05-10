import pytest
from env.Pacman.game import Directions, Configuration, Actions, GameState, GhostRules, PacmanRules, AgentState


def test_directions():
    assert Directions.LEFT['West'] == Directions.SOUTH
    assert Directions.RIGHT['West'] == Directions.NORTH
    assert Directions.REVERSE['West'] == Directions.EAST


def test_configuration():
    config = Configuration((0, 0), Directions.STOP)
    config2 = Configuration((0, 0), Directions.STOP)

    assert (0, 0) == config.getPosition()
    assert Directions.STOP == config.getDirection()
    assert config.isInteger()
    assert config == config2

    config = config.generateSuccessor([1, 0])
    assert config != config2

    assert str(config) == "(x,y)=(1, 0), East"


def test_actions(state: GameState):
    assert Actions.getActionWithIndex(0) == 'North'
    assert Actions.getActionWithIndex(3) == 'West'

    assert Actions.reverseDirection('North') == 'South'
    assert Actions.reverseDirection('South') == 'North'
    assert Actions.reverseDirection('East') == 'West'
    assert Actions.reverseDirection('West') == 'East'

    assert Actions.vectorToDirection((0, 1)) == 'North'
    assert Actions.vectorToDirection((0, -1)) == 'South'
    assert Actions.vectorToDirection((1, 0)) == 'East'
    assert Actions.vectorToDirection((-1, 0)) == 'West'
    assert Actions.vectorToDirection((0, 0)) == 'Stop'

    assert Actions.directionToVector('East') == (1, 0)

    assert Actions.getPossibleActions(state.agentStates[0], state.layout.walls) == ['South', 'East', 'West', 'Stop']


def test_ghost_rules(state: GameState):
    assert GhostRules.getLegalActions(state, 1) == ['East']
    assert GhostRules.getLegalActions(state, 2) == ['North', 'East', 'West']

    with pytest.raises(Exception) as ex:
        GhostRules.applyAction(state, "Stop", 1)

    assert 'Illegal ghost action Stop' in str(ex.value)

    GhostRules.applyAction(state, 'East', 1)
    assert state.getGhostState(1).getDirection() == 'East'

    ghost_state = state.getGhostState(1)
    ghost_state.scaredTimer = 1
    GhostRules.decrementTimer(state.getGhostState(1))
    assert ghost_state.scaredTimer == 0

    assert GhostRules.canKill((0, 0), (0, 0))
    GhostRules.checkDeath(state, 0)
    GhostRules.checkDeath(state, 1)
    ghost_state.configuration = state.getPacmanState().configuration
    ghost_state.scaredTimer = 1
    # kill ghost
    GhostRules.checkDeath(state, 0)
    # place ghost to start pos
    assert ghost_state.configuration == ghost_state.start

    # set ghost position to pacman position
    ghost_state.configuration = state.getPacmanState().configuration

    # kill pacman
    GhostRules.checkDeath(state, 1)
    assert state.isLose()


def test_pacman_rules(state: GameState):
    state.reset()
    assert PacmanRules.getLegalActions(state) == ['South', 'East', 'West', 'Stop']
    assert PacmanRules.getLegalActions(state, True) == ['South', 'East', 'West', 'Stop']

    # not in legal
    PacmanRules.applyAction(state, 'North')
    assert state.getPacmanDirection() == 'Stop'
    # legal
    PacmanRules.applyAction(state, 'East')
    assert state.getPacmanDirection() == 'East'
    assert state.scoreChange == 10


def test_agent_states():
    pacman = AgentState(Configuration((0, 0), 'Stop'), True)
    ghost = AgentState(Configuration((0, 1), 'Stop'), False)

    assert str(pacman) == "Pacman: (x,y)=(0, 0), Stop"
    assert str(ghost) == "Ghost: (x,y)=(0, 1), Stop"

    assert pacman != ghost

    assert pacman == pacman.copy()

    assert pacman.getPosition() == (0, 0)
    assert pacman.getDirection() == 'Stop'


def test_game_state(state: GameState):
    with pytest.raises(Exception) as ex:
        state.getGhostState(0)
    assert 'Invalid index' in str(ex.value)
