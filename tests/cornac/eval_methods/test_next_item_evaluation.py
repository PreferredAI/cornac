# Copyright 2023 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import unittest
import warnings

from cornac.eval_methods import NextItemEvaluation
from cornac.data import Reader
from cornac.models import SPop
from cornac.metrics import HitRatio, Recall


def _split_sids(dataset):
    """Raw session ids actually present in a built split."""
    inv = {v: k for k, v in dataset.sid_map.items()}
    return {inv[i] for i in dataset.sessions.keys()}


class TestNextItemEvaluation(unittest.TestCase):
    def setUp(self):
        self.data = Reader().read("./tests/sequence.txt", fmt="USIT", sep=" ")

    def test_from_splits(self):
        next_item_eval = NextItemEvaluation.from_splits(train_data=self.data[:50], test_data=self.data[50:], fmt="USIT")

        self.assertTrue(next_item_eval.train_set != None)
        self.assertTrue(next_item_eval.test_set != None)
        self.assertTrue(next_item_eval.val_set == None)
        self.assertTrue(next_item_eval.total_sessions == 16)

    def test_evaluate(self):
        next_item_eval = NextItemEvaluation.from_splits(train_data=self.data[:50], test_data=self.data[50:], fmt="USIT")
        result = next_item_eval.evaluate(
            SPop(), [HitRatio(k=2), Recall(k=2)], user_based=False
        )
        self.assertEqual(result[0].metric_avg_results.get('HitRatio@2'), 0)
        self.assertEqual(result[0].metric_avg_results.get('Recall@2'), 0)

        next_item_eval = NextItemEvaluation.from_splits(train_data=self.data[:50], test_data=self.data[50:], fmt="USIT")
        result = next_item_eval.evaluate(
            SPop(), [HitRatio(k=5), Recall(k=5)], user_based=True
        )
        self.assertEqual(result[0].metric_avg_results.get('HitRatio@5'), 2/3)
        self.assertEqual(result[0].metric_avg_results.get('Recall@5'), 2/3)

        next_item_eval = NextItemEvaluation.from_splits(train_data=self.data[:50], test_data=self.data[50:], fmt="USIT", mode="next")
        result = next_item_eval.evaluate(
            SPop(), [HitRatio(k=2), Recall(k=2)], user_based=False
        )

        self.assertEqual(result[0].metric_avg_results.get('HitRatio@2'), 1/8)
        self.assertEqual(result[0].metric_avg_results.get('Recall@2'), 1/8)

        next_item_eval = NextItemEvaluation.from_splits(train_data=self.data[:50], test_data=self.data[50:], fmt="USIT", mode="next")
        result = next_item_eval.evaluate(
            SPop(), [HitRatio(k=5), Recall(k=5)], user_based=True
        )
        self.assertEqual(result[0].metric_avg_results.get('HitRatio@5'), 3/4)
        self.assertEqual(result[0].metric_avg_results.get('Recall@5'), 3/4)

class TestFromTimestamps(unittest.TestCase):
    def setUp(self):
        # USIT: (user, session, item, timestamp). Rows interleaved across
        # sessions but chronological within each session. Session last-event
        # timestamps: s1=10, s2=40, s3=50, s4=70, s5=100, s6=130.
        self.usit = [
            ("u1", "s1", "a", 5),
            ("u1", "s1", "b", 10),
            ("u2", "s2", "a", 20),
            ("u2", "s2", "c", 40),
            ("u1", "s3", "b", 45),
            ("u1", "s3", "d", 50),
            ("u3", "s4", "a", 30),
            ("u3", "s4", "e", 70),
            ("u2", "s5", "c", 80),   # straddles test_timestamp=100 ...
            ("u2", "s5", "b", 100),  # ... but last event decides -> test
            ("u4", "s6", "f", 120),
            ("u4", "s6", "a", 130),
        ]

    def test_toy_assignment(self):
        m = NextItemEvaluation.from_timestamps(
            self.usit, test_timestamp=100, val_timestamp=50, fmt="USIT",
        )
        self.assertEqual(m.train_set.num_sessions, 2)
        self.assertEqual(m.val_set.num_sessions, 2)
        self.assertEqual(m.test_set.num_sessions, 2)
        self.assertEqual(_split_sids(m.train_set), {"s1", "s2"})
        self.assertEqual(_split_sids(m.val_set), {"s3", "s4"})
        # s5 straddles the cutoff (event at 80) yet lands in test by its last
        # event at 100.
        self.assertEqual(_split_sids(m.test_set), {"s5", "s6"})

    def test_boundary_equality(self):
        m = NextItemEvaluation.from_timestamps(
            self.usit, test_timestamp=100, val_timestamp=50, fmt="USIT",
        )
        # s3 last_ts == val_timestamp -> val; s5 last_ts == test_timestamp -> test
        self.assertIn("s3", _split_sids(m.val_set))
        self.assertIn("s5", _split_sids(m.test_set))

    def test_no_validation(self):
        m = NextItemEvaluation.from_timestamps(
            self.usit, test_timestamp=100, val_timestamp=None, fmt="USIT",
        )
        self.assertIsNone(m.val_set)
        # train absorbs the middle sessions (last_ts < 100)
        self.assertEqual(_split_sids(m.train_set), {"s1", "s2", "s3", "s4"})
        self.assertEqual(_split_sids(m.test_set), {"s5", "s6"})

    def test_val_ge_test_raises(self):
        with self.assertRaises(ValueError):
            NextItemEvaluation.from_timestamps(
                self.usit, test_timestamp=100, val_timestamp=100, fmt="USIT",
            )

    def test_empty_test_raises(self):
        with self.assertRaises(ValueError):
            NextItemEvaluation.from_timestamps(
                self.usit, test_timestamp=1000, val_timestamp=None, fmt="USIT",
            )

    def test_sit_format(self):
        # SIT: (session, item, timestamp) -- no user column.
        sit = [
            ("s1", "a", 5),
            ("s1", "b", 10),
            ("s2", "a", 20),
            ("s2", "c", 40),
            ("s3", "c", 80),
            ("s3", "b", 100),
        ]
        m = NextItemEvaluation.from_timestamps(
            sit, test_timestamp=100, val_timestamp=None, fmt="SIT",
        )
        self.assertEqual(_split_sids(m.train_set), {"s1", "s2"})
        self.assertEqual(_split_sids(m.test_set), {"s3"})

    def test_sorted_input_no_warning(self):
        # Chronological-within-session input must not trigger the task-01
        # sort-order warning from SequentialDataset.build.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            NextItemEvaluation.from_timestamps(
                self.usit, test_timestamp=100, val_timestamp=50, fmt="USIT",
            )


class TestLeaveLastOut(unittest.TestCase):
    def setUp(self):
        # UIRT: (user, item, rating, timestamp). Rows deliberately unsorted
        # within users. u1 has 4 interactions, u2 has 3, u3 only 2 (dropped).
        self.uirt = [
            ("u1", "c", 1.0, 30),
            ("u1", "a", 1.0, 10),
            ("u2", "b", 1.0, 25),
            ("u1", "d", 1.0, 40),
            ("u2", "a", 1.0, 5),
            ("u1", "b", 1.0, 20),
            ("u2", "c", 1.0, 45),
            ("u3", "a", 1.0, 15),
            ("u3", "b", 1.0, 35),
        ]

    def test_split_sizes_and_users(self):
        # exclude_unknowns=False isolates the split mechanics from the
        # unknown-item filtering done by from_splits.
        m = NextItemEvaluation.leave_last_out(self.uirt, exclude_unknowns=False)
        # kept users: u1 (n=4), u2 (n=3); u3 dropped (< 3 interactions).
        # cumulative splits: train sum(n-2)=3, val sum(n-1)=5, test sum(n)=7
        self.assertEqual(len(m.train_set.uir_tuple[0]), 3)
        self.assertEqual(len(m.val_set.uir_tuple[0]), 5)
        self.assertEqual(len(m.test_set.uir_tuple[0]), 7)
        self.assertEqual(_split_sids(m.train_set), {"u1", "u2"})
        self.assertEqual(_split_sids(m.val_set), {"u1", "u2"})
        self.assertEqual(_split_sids(m.test_set), {"u1", "u2"})

    def test_exclude_unknowns_default(self):
        # With the default exclude_unknowns=True, val/test items that never
        # appear in train (each user's held-out tail) are filtered out:
        # train has {a, b}; val keeps 4 of 5 rows, test keeps 4 of 7.
        m = NextItemEvaluation.leave_last_out(self.uirt)
        self.assertEqual(len(m.train_set.uir_tuple[0]), 3)
        self.assertEqual(len(m.val_set.uir_tuple[0]), 4)
        self.assertEqual(len(m.test_set.uir_tuple[0]), 4)

    def test_chronological_item_order(self):
        m = NextItemEvaluation.leave_last_out(self.uirt, exclude_unknowns=False)
        # u1's test session must be [a, b, c, d] (sorted by time, not input).
        test = m.test_set
        inv_iid = {v: k for k, v in test.iid_map.items()}
        sid = test.sid_map["u1"]
        items = [inv_iid[test.uir_tuple[1][idx]] for idx in test.sessions[sid]]
        self.assertEqual(items, ["a", "b", "c", "d"])

    def test_too_few_interactions_raises(self):
        with self.assertRaises(ValueError):
            NextItemEvaluation.leave_last_out(self.uirt[-2:])  # only u3

    def test_tied_timestamps_stable(self):
        # a and b share t=10: input order must be preserved on ties.
        uirt = [
            ("u1", "a", 1.0, 10),
            ("u1", "b", 1.0, 10),
            ("u1", "c", 1.0, 20),
        ]
        m = NextItemEvaluation.leave_last_out(uirt, exclude_unknowns=False)
        test = m.test_set
        inv_iid = {v: k for k, v in test.iid_map.items()}
        sid = test.sid_map["u1"]
        items = [inv_iid[test.uir_tuple[1][idx]] for idx in test.sessions[sid]]
        self.assertEqual(items, ["a", "b", "c"])

    def test_no_warning(self):
        # Output is sorted per session; the task-01 warning must not fire.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            NextItemEvaluation.leave_last_out(self.uirt)

    def test_mode_passthrough(self):
        m = NextItemEvaluation.leave_last_out(self.uirt, mode="next")
        self.assertEqual(m.mode, "next")


if __name__ == "__main__":
    unittest.main()
