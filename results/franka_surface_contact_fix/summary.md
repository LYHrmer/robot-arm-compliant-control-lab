# Smooth-contact repair validation

Full public development grid: 24 cases x 4 frames x 2 profiles = 192 rows.
Public development cases, NOT a new holdout. This compares contact-model profiles,
not controller superiority. No policy training, gain fitting or controller changes.
Setting both geoms' solimp d0 to zero changes the force-distance response as well as
continuity at contact onset. The model is not physically identified or hardware-validated.
Engineering repair checks: contact >=99%, raw-force RMSE <=2 N, full raw peak <=35 N,
and actuator saturation exactly 0%. These were not preregistered holdout gates.
There is no tangent gate; tangent tracking remains visible as a possible cost.
Contact and RMSE use evaluation_start; peaks and saturation use the full trial.

| Profile / arm | Contact min [%] | Contact median [%] | Raw RMSE median [N] | Raw RMSE P95 [N] | Peak max [N] | Peak P95 [N] | Tangent median [mm] | Worst saturation [%] | Repair passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy / world_safe_adaptive | 52.4 | 57.1333 | 11.0888 | 12.1069 | 33.9607 | 32.34 | 12.4673 | 0 | 0/24 |
| legacy / surface_exact | 52.4 | 56.6667 | 10.9164 | 11.9014 | 34.055 | 33.9738 | 11.8107 | 0 | 0/24 |
| legacy / surface_minus5 | 52.6 | 56.8 | 10.8991 | 11.8767 | 33.62 | 32.6605 | 11.7763 | 0 | 0/24 |
| legacy / surface_plus5 | 52.4667 | 56.8 | 10.9951 | 11.9763 | 32.9827 | 32.7654 | 12.0656 | 0 | 0/24 |
| smooth / world_safe_adaptive | 100 | 100 | 1.49791 | 1.54342 | 15.2446 | 15.2389 | 12.7418 | 0 | 24/24 |
| smooth / surface_exact | 100 | 100 | 0.181191 | 0.201121 | 12.6528 | 12.6498 | 11.8005 | 0 | 24/24 |
| smooth / surface_minus5 | 100 | 100 | 0.528842 | 0.576957 | 13.2926 | 13.2896 | 11.8253 | 0 | 24/24 |
| smooth / surface_plus5 | 100 | 100 | 0.367294 | 0.457261 | 12.832 | 12.8291 | 12.0344 | 0 | 24/24 |

paired_deltas.csv records smooth minus legacy for each identical case and frame.
All selected legacy runs first reproduce the archived metrics within absolute 1e-10.
The representative image and four smooth input traces use preselected case 16 only;
they are not the aggregate result. Legacy trace references are retained by hash.
Input replay reproduces controller mathematics, not closed-loop robot dynamics.
Geometry uses the true wall plane: gap >0 means separation, gap <0 penetration.
max_geometric_separation_um measures task-period separation; optional mean true
tangent force and median ||Ft||/Fn (Fn >0.5 N) use recorded contact loads.
Empty auxiliary values mean NA, not zero force or verified absence of separation.
Changing this contact model does not establish physical collision safety.
