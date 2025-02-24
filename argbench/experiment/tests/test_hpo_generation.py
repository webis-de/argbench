import unittest

from unittest import TestCase
from ..run import Runner

class HPOUnitTest(TestCase):
    def test_hpo(self):


        config_list = ["/home/yamen/projects/task-specific-argument-mining-and-generation/src/experiment/configs/complete_leave_one_out_ajjour17_hpo.json"]
        config = RunConfig.from_file(config_list)


