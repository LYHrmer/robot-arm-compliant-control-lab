# Classical tangential-compensation comparison

Full public grid: 24/24 cases x 3 methods; selected indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23].
Public development data, NOT a new holdout. The smooth contact model is fixed;
this does not identify physical contact parameters or establish real-robot safety.
No tuning: integral gain 800, correction cap 6 N, nominal friction 0.45, velocity scale 0.005 m/s.
Engineering criteria: main median tangent RMSE <= half baseline; every case contact >=99%,
raw peak <=35 N, saturation exactly 0%, and paired force-RMSE increase <=0.2 N.
These are development checks, not preregistered holdout gates. Failures are retained.
Force/tangent RMSE and contact use evaluation_start; peak, projection and saturation use the full trial.
Penetration is max(0, -signed gap); maximum uses the full trial, median uses evaluation_start.
Requested correction is before torque projection; input replay is controller mathematics, not plant replay.

## Main fixed 24-case grid

| Method | Cases | Tangent median [mm] | Force RMSE median [N] | Peak max [N] | Contact min [%] | Sat max [%] | Penetration max [mm] | Projection max [%] | Case passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 24 | 11.8005 | 0.181191 | 12.6528 | 100 | 0 | 0.851101 | 0 | 24/24 |
| integral | 24 | 8.09288 | 0.191857 | 12.6594 | 100 | 0 | 0.851086 | 0 | 24/24 |
| friction | 24 | 1.88541 | 0.168516 | 12.6528 | 100 | 0 | 0.851101 | 0 | 24/24 |

integral: tangent median ratio=0.68581; tangent criterion=False; all case checks=True; overall=False.

friction: tangent median ratio=0.159774; tangent criterion=True; all case checks=True; overall=True.

## Separate long-duration friction-mismatch diagnostic

Original case 16, duration 12 s; both actual tool/wall friction coefficients change together.
Controller nominal friction remains 0.45. These 9 runs are NOT pooled with the main grid.
| True friction | Method | Tangent [mm] | Force RMSE [N] | Contact [%] | Peak [N] | Sat [%] | Penetration max [mm] | Failed checks |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0.25 | baseline | 7.50357 | 0.111934 | 100 | 12.6189 | 0 | 0.789615 | none |
| 0.25 | integral | 5.04419 | 0.116584 | 100 | 12.6171 | 0 | 0.7896 | none |
| 0.25 | friction | 3.87527 | 0.100204 | 100 | 12.7069 | 0 | 0.789615 | none |
| 0.45 | baseline | 11.7154 | 0.115252 | 100 | 12.6207 | 0 | 0.789633 | none |
| 0.45 | integral | 8.10798 | 0.123388 | 100 | 12.6192 | 0 | 0.789619 | none |
| 0.45 | friction | 1.70517 | 0.110468 | 100 | 12.6207 | 0 | 0.789633 | none |
| 0.65 | baseline | 15.5901 | 0.118258 | 100 | 12.6207 | 0 | 0.789633 | none |
| 0.65 | integral | 10.0522 | 0.127201 | 100 | 12.6625 | 0 | 0.789619 | none |
| 0.65 | friction | 6.48408 | 0.119006 | 100 | 12.6207 | 0 | 0.789633 | none |

Both paired CSVs use method minus matching baseline; do not reward a missing-contact peak.
Only main case 16 (three methods) and long case 16 / friction 0.45 / friction method retain full traces.
Other rows retain configurations and metrics only. The representative image is not an aggregate result.
