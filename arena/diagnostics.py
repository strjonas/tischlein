"""Judge-side telemetry. Never passed into a controller or policy observation."""
import numpy as np


class Trace:
    def __init__(self, world, scenario):
        self.stages = {}
        self.samples = []
        self.last_stage = None
        self.last_sample = -100
        self.origins = {name: world.obj_pos(name).copy() for name in scenario.objs}
        self.milestones = dict(seen=False, lifted=False, transported=False)

    def step(self, w, c):
        stage = c.stage
        for name, start in self.origins.items():
            self.milestones["seen"] |= name in c.mem and c.mem[name][3] not in ("prior", "memory")
            p, ee = w.obj_pos(name), w.d.site_xpos[w.ee]
            in_hand = np.linalg.norm(p-ee) < .18
            self.milestones["lifted"] |= bool(p[2] > start[2]+.03 and in_hand)
            self.milestones["transported"] |= bool(self.milestones["lifted"] and np.linalg.norm(p[:2]-start[:2]) > .5 and in_hand)
        row = self.stages.setdefault(stage, dict(seconds=0.0, max_person_N=0.0, max_tilt_deg=0.0,
                                                 max_tracking_error=0.0, max_motor_fraction=0.0))
        row["seconds"] += 0.02
        row["max_person_N"] = max(row["max_person_N"], w.mon["max_person"])
        row["max_tilt_deg"] = max(row["max_tilt_deg"], w.mon["max_tilt"])
        row["max_tracking_error"] = max(row["max_tracking_error"], float(np.max(np.abs(w.true_q()-c.q_cmd))))
        if c.obs is not None:
            frac = float(np.max(np.abs(c.obs.tau)/np.array([a.force for a in c.r.chain])))
            row["max_motor_fraction"] = max(row["max_motor_fraction"], frac)
        if stage != self.last_stage or w.t-self.last_sample >= 5:
            sample = dict(t=round(w.t, 2), stage=stage, base=list(w.base_pose()),
                          estimated_base=list(c.pose()), ee=w.d.site_xpos[w.ee].tolist(),
                          objects={k: dict(pos=w.obj_pos(k).tolist(), up=float(w.obj_mat(k)[2, 2])) for k in w.obj},
                          grip=round(w.true_grip(), 4), spill=w.mon["spill"], held_claim=c.held)
            sample.update(q=w.true_q().tolist(), q_command=c.q_cmd.tolist(), grip_force=round(float(c.obs.grip_f), 2) if c.obs else 0)
            self.samples.append(sample)
            self.last_sample = w.t
        self.last_stage = stage

    def result(self, w, c):
        for row in self.stages.values():
            for k in row:
                row[k] = round(row[k], 4)
        return dict(stages=self.stages, samples=self.samples, controller_failure=c.failure,
                    events=c.events, terminal_stage=c.stage,
                    contacts=sorted(w.mon.get("contact_peaks", {}).items(), key=lambda kv: -kv[1])[:12], milestones=self.milestones)
