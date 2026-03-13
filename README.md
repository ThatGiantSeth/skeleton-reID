# Skeleton Tracking and Reidentification with RGB-D cameras
By Aubrey McKinney and Seth Myers @ ODU Vision Lab

Senior Design Project 2025-2026

To install and run: [View Wiki](https://github.com/ThatGiantSeth/skeleton-reID/wiki)

## Project Checklist:
- [x] Implement k-fold validation
- [ ] Write training and usage documentation
- [ ] Create demo video
- [x] add argparse so that model and parameters can be changed at runtime
- [x] add some QoL features to recording script
- [x] Figure out how to match numbers to names
- [x] Fix GUI only locking onto a person once occasionally (IMPORTANT CAVEAT: Subsequent lock-on takes a few seconds because NiTE waits for some time before unloading a user after they leave the frame)
- [ ] Fix GUI freezing on startup occasionally (likely due to initialization issue with camera)
- [ ] Compartmentalize client code so that OpenNI can be swapped out for a different architecture in the future
- [ ] ~~Fix memory leak caused by NiTE2 Python bindings~~ - not possible within the scope of this project, requires editing NiTE itself