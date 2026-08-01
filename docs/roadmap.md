# Roadmap

## Milestone 1 — hardware validation

- compile `legacy-capture-worker` on the target Windows machine
- confirm the USB 3 controller sustains synchronized depth and color
- validate depth-to-color registration with a checkerboard scene
- measure practical capture rate at 5, 10 and 15 fps

## Milestone 2 — first real room

- record one short phase with corners and furniture
- tune RGB-D odometry thresholds for Kinect v2 noise
- record a second overlapping phase and validate global registration
- compare 10, 15 and 25 mm output spacing

## Milestone 3 — alignment tools

- add a three-correspondence manual alignment screen
- visualize registration fitness and overlap
- support fixed ArUco boards as cross-session anchors
- permit phase reordering and exclusion

## Milestone 4 — Unity output

- RGB-keyframe-reprojected triangle mesh and texture-atlas export (implemented)
- connected-component cleanup and quadric decimation
- floor alignment and origin selection
- GLB packaging and simplified collider export

## Milestone 5 — distribution

- bundle both workers into the Windows installer
- add capture recovery after application interruption
- add project chooser, rename and archive actions
- hardware diagnostics for Kinect runtime, power adapter and USB controller
