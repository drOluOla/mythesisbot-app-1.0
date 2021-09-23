run-actions:
	python -m rasa_sdk.endpoint --actions thesisbot_actions

run-thesisbot:
	make run-actions&
	python thesisbot_runapp.py
