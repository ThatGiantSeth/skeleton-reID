# Skeleton Tracking and Reidentification with RGB-D cameras
## Old Dominion University Vision Lab
### ECE 481W/482 Senior Design

## Project Checklist:
- [x] Implement k-fold validation
- [ ] Write training and usage documentation
- [ ] Create demo video
- [ ] Fix issue where it will not track a user more than once per run
- [x] add argparse so that model and parameters can be changed at runtime
- [x] add some QoL features to recording script
- [x] Figure out how to match numbers to names
- [ ] Fix GUI only locking onto a person once occasionally
- [ ] Fix GUI freezing on startup occasionally (likely due to initialization issue with camera)
- [ ] Compartmentalize client code so that OpenNI can be swapped out for a different architecture in the future
- [ ] ~~Fix memory leak caused by NiTE2 Python bindings~~ - not possible within the scope of this project, requires editing NiTE itself