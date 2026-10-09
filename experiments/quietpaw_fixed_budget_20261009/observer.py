"""Evaluation-only observer reused from the Oct09 deployment comparison.

Original source: legged_gym/logs/quietpaw_speed_only8000_comparison_20261009/
evaluation_program.py. Training never uses this observer.
"""

import torch

class PhysicalObserver:
    """Refresh only; no changes to actions, forces, dynamics, or reward."""
    def __init__(self, env, gym):
        self.env, self.gym = env, gym
        self.n, self.dev = env.num_envs, env.device
        self.dt = env.sim_params.dt
        self.window = 8  # 40 ms at 200 Hz, fixed before looking at results
        self.tick = 0
        self.active = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        f = env.contact_forces[:, env.feet_indices].clone()
        self.previous_force = f
        self.previous_contact = f.norm(dim=-1) > 1.
        self.previous_foot_vz = env.feet_vel[..., 2].clone()
        self.previous_base_v = env.root_states[:, 7:10].clone()
        shape = (self.n, 4, self.window)
        self.remaining = torch.zeros(shape, dtype=torch.long, device=self.dev)
        self.peak_fz = torch.zeros(shape, device=self.dev)
        self.peak_f3d = torch.zeros(shape, device=self.dev)
        self.peak_df = torch.zeros(shape, device=self.dev)
        self.impact_speed = torch.zeros(shape, device=self.dev)
        self.sums = {k:torch.zeros(self.n, device=self.dev) for k in
            ['samples', 'contact_samples', 'fz_sum', 'fz_sq_sum', 'a_sq_sum',
             'a_pre_push_sq_sum', 'pre_push_samples', 'touchdowns',
             'completed_events', 'event_fz_sum', 'event_f3d_sum', 'event_df_sum',
             'hard_force_events', 'hard_velocity_events', 'censored_events']}
        self.event_values = {k:[] for k in ['fz', 'f3d', 'df', 'speed']}
        self.event_env_ids = []

    def sample(self):
        e = self.env
        # P-controller reads these joint tensors, not root/foot/contact caches.
        # Assert every sample, before any additional refresh is exposed.
        pd_pos, pd_vel = e.dof_pos.clone(), e.dof_vel.clone()
        self.gym.refresh_net_contact_force_tensor(e.sim)
        self.gym.refresh_actor_root_state_tensor(e.sim)
        self.gym.refresh_rigid_body_state_tensor(e.sim)
        if not torch.equal(pd_pos, e.dof_pos) or not torch.equal(pd_vel, e.dof_vel):
            raise RuntimeError("Observer refresh changed PD joint-state inputs")
        f = e.contact_forces[:, e.feet_indices].clone()
        foot_vz = e.rigid_body_states.view(self.n, e.num_bodies, 13)[:, e.feet_indices, 9].clone()
        base_v = e.root_states[:, 7:10].clone()
        mag = f.norm(dim=-1)
        contact = mag > 1.
        td = contact & ~self.previous_contact & self.active[:, None]
        fz = f[..., 2].clamp_min(0)
        df = (f-self.previous_force).norm(dim=-1)/self.dt
        a_g_sq = ((base_v-self.previous_base_v)/self.dt/9.81).square().sum(-1)
        active = self.active.float()
        cm = contact & self.active[:, None]
        self.sums['samples'] += active
        self.sums['contact_samples'] += cm.sum(-1)
        self.sums['fz_sum'] += (fz*cm).sum(-1)
        self.sums['fz_sq_sum'] += (fz.square()*cm).sum(-1)
        self.sums['a_sq_sum'] += a_g_sq*active
        if e.common_step_counter < e.cfg.domain_rand.push_interval:
            self.sums['a_pre_push_sq_sum'] += a_g_sq*active
            self.sums['pre_push_samples'] += active
        self.sums['touchdowns'] += td.sum(-1)
        slot = self.tick % self.window
        self.remaining[:, :, slot] = td.long()*self.window
        self.peak_fz[:, :, slot] = 0.
        self.peak_f3d[:, :, slot] = 0.
        self.peak_df[:, :, slot] = 0.
        self.impact_speed[:, :, slot] = (-self.previous_foot_vz).clamp_min(0)*td
        running = (self.remaining > 0) & self.active[:, None, None]
        self.peak_fz = torch.where(running, torch.maximum(self.peak_fz, fz[..., None]), self.peak_fz)
        self.peak_f3d = torch.where(running, torch.maximum(self.peak_f3d, mag[..., None]), self.peak_f3d)
        self.peak_df = torch.where(running, torch.maximum(self.peak_df, df[..., None]), self.peak_df)
        self.remaining -= running.long()
        finished = running & (self.remaining == 0)
        self.sums['completed_events'] += finished.sum((1,2))
        self.sums['event_fz_sum'] += (self.peak_fz*finished).sum((1,2))
        self.sums['event_f3d_sum'] += (self.peak_f3d*finished).sum((1,2))
        self.sums['event_df_sum'] += (self.peak_df*finished).sum((1,2))
        self.sums['hard_force_events'] += (finished & (self.peak_f3d >= 100.)).sum((1,2))
        self.sums['hard_velocity_events'] += (finished & (self.impact_speed >= .5)).sum((1,2))
        indices = finished.nonzero(as_tuple=True)
        self.event_env_ids.append(indices[0])
        for name, data in [('fz', self.peak_fz), ('f3d', self.peak_f3d), ('df', self.peak_df), ('speed', self.impact_speed)]:
            self.event_values[name].append(data[indices])
        self.previous_force, self.previous_contact = f, contact
        self.previous_foot_vz, self.previous_base_v = foot_vz, base_v
        self.tick += 1

    def finish_control(self, dones, last=False):
        ending = self.active & (dones.bool() | last)
        self.sums['censored_events'] += ((self.remaining > 0) & ending[:,None,None]).sum((1,2))
        self.remaining[ending] = 0
        self.active[ending] = False

    def result(self):
        s = self.sums
        duration = s['samples']*self.dt
        contact_count = s['contact_samples'].clamp_min(1)
        fzmean = s['fz_sum']/contact_count
        fzstd = (s['fz_sq_sum']/contact_count-fzmean.square()).clamp_min(0).sqrt()
        per_env = dict(fz_contact_mean_N=fzmean,
            fz_contact_std_N=fzstd,
            base_acc_rms_g=(s['a_sq_sum']/s['samples'].clamp_min(1)).sqrt(),
            base_acc_pre_push_rms_g=(s['a_pre_push_sq_sum']/s['pre_push_samples'].clamp_min(1)).sqrt(),
            touchdown_rate_hz=s['touchdowns']/duration.clamp_min(self.dt),
            hard_force_rate_hz=s['hard_force_events']/duration.clamp_min(self.dt),
            hard_velocity_rate_hz=s['hard_velocity_events']/duration.clamp_min(self.dt),
            touchdown_count=s['touchdowns'], completed_events=s['completed_events'],
            censored_events=s['censored_events'])
        summary = {k:v.mean().item() for k,v in per_env.items()}
        ids = torch.cat(self.event_env_ids)
        for name in ['fz', 'f3d', 'df', 'speed']:
            values = torch.cat(self.event_values[name])
            prefix = 'event_'+name+'_peak' if name != 'speed' else 'event_impact_speed'
            if not len(values):
                raise RuntimeError('No completed touchdown windows')
            summary[prefix+'_mean'] = values.mean().item()
            summary[prefix+'_p95'] = torch.quantile(values,.95).item()
            summary[prefix+'_max'] = values.max().item()
            sums = torch.zeros(self.n,device=self.dev).scatter_add_(0,ids,values)
            means = sums/s['completed_events'].clamp_min(1)
            per_env[prefix+'_mean'] = means
        summary['hard_force_fraction'] = (s['hard_force_events'].sum()/s['completed_events'].sum().clamp_min(1)).item()
        summary['hard_velocity_fraction'] = (s['hard_velocity_events'].sum()/s['completed_events'].sum().clamp_min(1)).item()
        return summary,{k:v.cpu().tolist() for k,v in per_env.items()}

class GymProxy:
    def __init__(self, gym, observer):
        self.original, self.observer = gym, observer
    def __getattr__(self, name):
        return getattr(self.original,name)
    def refresh_dof_state_tensor(self, sim):
        result = self.original.refresh_dof_state_tensor(sim)
        self.observer.sample()
        return result

