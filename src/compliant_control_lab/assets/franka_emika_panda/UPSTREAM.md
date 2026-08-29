# Upstream provenance

The files in this directory were copied from
[`google-deepmind/mujoco_menagerie/franka_emika_panda`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda)
at commit `da76818e269b82289eba39808e2fb91d679d6994`.

They are distributed under the included Apache-2.0 `LICENSE`.

`panda_torque.xml` is a local derivative of `panda.xml`. It changes the seven arm actuators from
position control to torque control and adds a rigid spherical contact tool and `ee_site`.
