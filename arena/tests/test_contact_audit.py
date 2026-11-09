"""A gripping hand does not make all hand contacts intentional."""
import unittest

from arena.contact_audit import clean


class Contacts(unittest.TestCase):
    def test_furniture_contact_by_a_jaw_invalidates_clean_claim(self):
        row = dict(ok=True, max_bump=0, collision_free=True,
                   diagnostics=dict(contacts=[("arm_jaw_a / bedside", 44.8)]))
        self.assertFalse(clean(row))

    def test_object_gripping_force_is_not_a_furniture_collision(self):
        row = dict(ok=True, max_bump=0,
                   diagnostics=dict(contacts=[("obj_pill_bottle_0 / pad_a", 43.4)]))
        self.assertTrue(clean(row))

    def test_truncated_high_force_contacts_do_not_certify_an_unknown(self):
        row = dict(ok=True, max_bump=0,
                   diagnostics=dict(contacts=[(f"obj_box_{i} / pad_a", 100-i) for i in range(12)]))
        self.assertFalse(clean(row))


if __name__ == "__main__":
    unittest.main()
