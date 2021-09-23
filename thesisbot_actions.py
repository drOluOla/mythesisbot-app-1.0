# -*- coding: utf-8 -*-

import logging
import re
import yaml

from typing import Text, Dict, Any, List
import json

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted, SlotSet, ConversationPaused

logger = logging.getLogger(__name__)

stream = open('thesisbot_config.yml', 'r')
thesis_kwargs = yaml.safe_load(stream)


class ActionChitchat(Action):
    """Returns the chitchat utterance dependent on the intent"""

    def name(self):
        return "action_chitchat"

    def run(self, dispatcher, tracker, domain):
        intent = tracker.latest_message['intent'].get('name')

        # retrieve the correct chitchat utterance dependent on the intent
        if intent in ['ask_builder', 'ask_weather', 'ask_howdoing', 'ask_whatspossible', 'ask_whatisthesisbot', 'ask_isbot',
                      'ask_howold', 'ask_languagesbot', 'ask_restaurant', 'ask_time', 'ask_wherefrom', 'ask_whoami',
                      'handleinsult', 'nicetomeeyou', 'telljoke', 'ask_whatismyname', 'ask_whoisit',
                      'give_feedback', 'other_possibilities', 'researcher_handoff', 'thankyou', 'bye']:
            dispatcher.utter_template('utter_' + intent, tracker,  **thesis_kwargs)
        elif intent in ['canthelp', 'out_of_scope']:
            dispatcher.utter_template('utter_' + intent, tracker)
            dispatcher.utter_template('utter_how_to_get_started_more', tracker,  **thesis_kwargs)
        elif intent == 'how_to_get_started':
            dispatcher.utter_template('utter_how_to_get_started', tracker,  **thesis_kwargs)
            dispatcher.utter_template('utter_type_message', tracker)
        elif intent == 'mood_approve':
            dispatcher.utter_template('utter_awesome', tracker) 
            dispatcher.utter_template('utter_type_message', tracker)
        elif intent == 'mood_disapprove':
            dispatcher.utter_template('utter_canthelp', tracker) 
            dispatcher.utter_template('utter_ask_know_more', tracker, **thesis_kwargs)
        return []


class ActionFaqs(Action):
    """Returns the faqs utterance dependent on the intent"""

    def name(self):
        return "action_faqs"

    def run(self, dispatcher, tracker, domain):
        intent = tracker.latest_message['intent'].get('name')

        # retrieve the correct utterance dependent on the intent
        if intent in ['ask_faq_researcher1', 'ask_faq_researcher2', 'ask_faq_supervisor', 'ask_faq_examiners', 'ask_faq_university',
                      'ask_faq_vivadate', 'ask_faq_graduationdate', 'ask_faq_graduationvenue', 'ask_faq_researchduratioin',
                      'ask_faq_researchcost', 'ask_faq_how_botwasbuild', 'ask_faq_languages', 'ask_faq_toobroad', 'ask_faq_questions_to_ask']:
            dispatcher.utter_template('utter_' + intent, tracker, **thesis_kwargs)
        return []


class ActionResearch(Action):
    """Returns the research utterance dependent on the intent"""

    def name(self):
        return "action_research"

    def run(self, dispatcher, tracker, domain):
        intent = tracker.latest_message['intent'].get('name')

        if intent in ['ask_research_title','ask_summary','ask_dumbed_down_summary','ask_research_area','ask_main_idea','ask_research_aim', 
                      'ask_research_motivation','ask_most_interesting_aspect','ask_most_enjoyed_aspect', 
                      'ask_suprised','ask_change_progression','ask_phd_lessons','ask_newresearcher_advice', 
                      'ask_problem_identification','ask_strongest_influence','ask_most_important_papers', 
                      'ask_primary_definition','ask_secondary_definition','ask_most_recent_developments', 
                      'ask_key_research_decisions','ask_experimental_design','ask_research_methodology','ask_research_methodology_options', 
                      'ask_research_methodology_options_advantage','ask_ethical_issues','ask_dataset','ask_evaluation', 
                      'ask_evaluation_correct','ask_strongest','ask_weakest','ask_improvement','ask_summarise_finding','ask_finding_literature_relation',
                      'ask_empirical_practical_implications','ask_phd_interests', 
                      'ask_phd_publication','ask_research_question1','ask_research_question2','ask_research_question3',
                      'ask_research_contribution1','ask_research_contribution2','ask_research_contribution3',
                      'ask_research_futurework1','ask_research_futurework2','ask_research_futurework3', 'ask_researcher_status']:
            dispatcher.utter_template('utter_' + intent, tracker, **thesis_kwargs)       
        elif intent in ['ask_research_questions','ask_research_contributions','ask_research_futureworks']:  
            key_count = []
            for key, value in thesis_kwargs.items():
                key_string = key[:-1]
                if re.search('{}'.format(re.escape(key_string)),intent):
                    key_count.append(key)
                             
            #Adjust intent to match key in kwargs dictionary
            for i in range(1, len(key_count)+1):
                repl_key = intent.replace('ask_', '')[:-1] + str(i)
                dispatcher.utter_template('utter_ask_' + repl_key, tracker, **thesis_kwargs)

        return []


class ActionGreetUser(Action):
    """Greets the user"""
    
    def name(self):
        return "action_greet_user"

    def run(self, dispatcher, tracker, domain):
        intent = 'greet'
        dispatcher.utter_template("utter_" + intent, tracker, **thesis_kwargs)
        dispatcher.utter_template("utter_how_to_get_started", tracker, **thesis_kwargs)
        dispatcher.utter_template("utter_type_message", tracker)

        return []


class ActionStoreFeedback(Action):
    """Stores the feedback text in a slot"""

    def name(self):
        return "action_store_feedback"

    def run(self, dispatcher, tracker, domain):

        feedback_text = tracker.latest_message.get('text')

        return [SlotSet('feedback', feedback_text)]

    
class ActionDefaultAskAffirmation(Action):
    """Asks for an affirmation of the intent if NLU threshold is not met."""

    def name(self) -> Text:
        return "action_default_ask_affirmation"

    def __init__(self) -> None:
        import pandas as pd

        self.intent_mappings = pd.read_csv("data/"
                                           "intent_description_mapping.csv")
        self.intent_mappings.fillna("", inplace=True)
        self.intent_mappings.entities = self.intent_mappings.entities.map(
            lambda entities: {e.strip() for e in entities.split(',')})

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
            ) -> List['Event']:

        intent_ranking = tracker.latest_message.get('intent_ranking', [])
        if len(intent_ranking) > 1:
            diff_intent_confidence = (intent_ranking[0].get("confidence") -
                                      intent_ranking[1].get("confidence"))
            if diff_intent_confidence < 0.2:
                intent_ranking = intent_ranking[:2]
            else:
                intent_ranking = intent_ranking[:1]
        first_intent_names = [intent.get('name', '')
                              for intent in intent_ranking
                              if intent.get('name', '') != 'out_of_scope']

        message_title = ("Sorry, I'm not sure I've understood "
                         "you correctly 🤔. Do want to...")

        entities = tracker.latest_message.get("entities", [])
        entities = {e["entity"]: e["value"] for e in entities}

        entities_json = json.dumps(entities)

        buttons = []
        for intent in first_intent_names:
            logger.debug(intent)
            logger.debug(entities)
            buttons.append({'title': self.get_button_title(intent, entities),
                            'payload': '/{}{}'.format(intent,
                                                      entities_json)})

        buttons.append({'title': 'See Question Examples',
                        'payload': '/ask_faq_questions_to_ask'})

        buttons.append({'title': 'None of the Above',
                        'payload': '/out_of_scope'})

        dispatcher.utter_button_message(message_title, buttons=buttons)

        return []

    def get_button_title(self, intent: Text, entities: Dict[Text, Text]
                         ) -> Text:
        default_utterance_query = self.intent_mappings.intent == intent
        utterance_query = (
                (self.intent_mappings.entities == entities.keys()) &
                default_utterance_query)

        utterances = self.intent_mappings[utterance_query].button.tolist()

        if len(utterances) > 0:
            button_title = utterances[0]
        else:
            utterances = (
                self.intent_mappings[default_utterance_query]
                    .button.tolist())
            button_title = utterances[0] if len(utterances) > 0 else intent

        return button_title.format(**entities)


class ActionDefaultFallback(Action):

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
            ) -> List['Event']:

        # Fallback caused by TwoStageFallbackPolicy
        if (len(tracker.events) >= 4 and
                tracker.events[-4].get('name') ==
                'action_default_ask_affirmation'):

            dispatcher.utter_template('utter_restart_with_button', tracker)

            return [SlotSet('feedback_value', 'negative'),
                    ConversationPaused()]

        # Fallback caused by Core
        else:
            dispatcher.utter_template('utter_default', tracker)
            return [UserUtteranceReverted()]
