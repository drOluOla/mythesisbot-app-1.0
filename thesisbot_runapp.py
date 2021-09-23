import logging
import rasa
import yaml
import os
import asyncio

from rasa.core.agent import Agent
# from rasa.core.policies.keras_policy import KerasPolicy
from rasa.core.policies.memoization import AugmentedMemoizationPolicy
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.ted_policy import TEDPolicy

from rasa.core.utils import AvailableEndpoints, EndpointConfig
from rasa.core.run import serve_application
from rasa.core import config
from rasa.core.interpreter import RasaNLUInterpreter

from rasa.shared.nlu.training_data.loading import load_data
from rasa.nlu import config
from rasa.nlu.model import Trainer
from rasa.nlu.model import Metadata, Interpreter

from rasa.core.channels.socketio import SocketIOInput
from rasa.core.channels.facebook import FacebookInput
from rasa.core.utils import read_endpoints_from_path

logger = logging.getLogger(__name__)


def train_thesisbot_dialogue(domain_file='thesisbot_domain.yml', model_path='./models/dialogue',
                             training_data_file='./data/core'):
    #agent = Agent(domain_file, policies=[MemoizationPolicy(max_history=3), KerasPolicy(max_history=3, epochs=50), FallbackPolicy(fallback_action_name="action_default_ask_affirmation", nlu_threshold=0.8, core_threshold=0.3)])
    agent = Agent(domain_file, policies=[AugmentedMemoizationPolicy(max_history=3), TEDPolicy(max_history=3, epochs=50),
                                         FallbackPolicy(fallback_action_name="action_default_ask_affirmation",
                                                        nlu_threshold=0.8, core_threshold=0.3)])
    data = asyncio.run(agent.load_data(training_data_file, augmentation_factor=0))
    agent.train(data)
    agent.persist(model_path)


def train_thesisbot_nlu(data, configs, model_directory):
    training_data = load_data(data)
    training_agent = Trainer(config.load(configs))
    training_agent.train(training_data)
    training_agent.persist(model_directory, fixed_model_name='current')


def run_thesisbot():
    core_model_path = './models/dialogue'

    nlu_interpreter = RasaNLUInterpreter('./models/current')
    action_endpoint_webhook = EndpointConfig(url="http://localhost:5055/webhook")

    #avail_endpoint = AvailableEndpoints(action_endpoint_webhook)
    avail_endpoint = read_endpoints_from_path(endpoints_path="endpoint.yml")

    #action_endpoint_webhook = EndpointConfig(url="http://localhost:5055/webhook")
    #agent_soc = Agent.load(core_model_path, interpreter=nlu_interpreter, action_endpoint=action_endpoint_webhook)
    #agent_fbk = Agent.load(core_model_path, interpreter=nlu_interpreter, action_endpoint=action_endpoint_webhook)
    agent_cmd = Agent.load(core_model_path, interpreter=nlu_interpreter, action_endpoint=action_endpoint_webhook)


    rasa.core.run.serve_application(core_model_path, channel='socketio', credentials='credentials.yml', endpoints=avail_endpoint, port=5004, cors="*")


    # return agent_soc, agent_fbk


if __name__ == '__main__':
    #train_thesisbot_dialogue()
    #train_thesisbot_nlu('./data/nlu/thesisbot_nlu.md', 'nlu_tensorflow.yml', './models')

    run_thesisbot()
