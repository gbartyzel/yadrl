import json
import logging
import os
import re
from copy import deepcopy
from typing import Any

import jsonschema
import torch


class CheckpointManager(object):
    _SUMMARY_SCHEMA = {
        "type": "object",
        "properties": {
            "agent_type": {"type": "string"},
            "recent": {"type": "string"},
            "previous": {"type": "object"}
        },
        "required": ["agent_type", "recent"]

    }

    _SUMMARY_PROTO = {
        "agent_type": "",
        "recent": "",
        "previous": {}
    }

    def __init__(self,
                 agent_type: str,
                 checkpoint_path: str,
                 history_length: int = 5):
        self._agent_type = agent_type
        self._log_dir = checkpoint_path
        self._checkpoint = os.path.join(checkpoint_path, 'checkpoint_{}.pth')
        self._summary = os.path.join(checkpoint_path, 'checkpoint_summary.json')
        self._history_length = history_length

        self._create_checkpoint_summary()

    def save(self, state_dicts: Any, step: int):
        torch.save(state_dicts, self._checkpoint.format(step))
        self._update_summary(self._checkpoint.format(step))

    def load(self):
        checkpoint = self._get_recent_checkpoint()
        if os.path.isfile(checkpoint):
            model = torch.load(checkpoint)
            step = self._get_checkpoint_step(checkpoint)
            logging.info('Checkpoint {} loaded'.format(step))
            return model
        logging.info('Checkpoint not found')
        return None

    def _create_checkpoint_summary(self):
        self._create_summary_directory(os.path.split(self._summary)[0])
        if not os.path.isfile(self._summary):
            with open(self._summary, 'w+') as summary_file:
                proto = deepcopy(self._SUMMARY_PROTO)
                proto['agent_type'] = self._agent_type
                json.dump(proto, summary_file)
            return
        self._validate_summary()

    def _validate_summary(self):
        with open(self._summary, 'r') as summary_file:
            summary = json.load(summary_file)
            jsonschema.validate(instance=summary, schema=self._SUMMARY_SCHEMA)
            if not summary['agent_type'] == self._agent_type:
                raise ValueError

    @staticmethod
    def _create_summary_directory(checkpoint_dir: str):
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def _update_summary(self, checkpoint: str):
        with open(self._summary, 'r') as summary_file:
            summary = json.load(summary_file)

        summary['agent_type'] = self._agent_type
        summary['recent'] = checkpoint
        prev_len = len(summary['previous'])

        for i in range(min(prev_len, self._history_length - 1)):
            temp = summary['previous'][str(i + 1)]
            summary['previous'][str(i + 2)] = temp
        summary['previous']['1'] = checkpoint

        with open(self._summary, 'w') as summary_file:
            json.dump(summary, summary_file)

    def _get_recent_checkpoint(self) -> str:
        with open(self._summary) as summary_file:
            summary = json.load(summary_file)
            return summary['recent']

    def _get_checkpoint(self, checkpoint_num) -> str:
        with open(self._summary) as summary_file:
            summary = json.load(summary_file)
            return summary['previous'][str(checkpoint_num)]

    @staticmethod
    def _get_checkpoint_step(checkpoint) -> int:
        _, checkpoint_file = os.path.split(checkpoint)
        _, step, _ = re.split('[_.]', checkpoint_file)
        return step
