import launch

if not launch.is_installed("facefinder==0.1.4"):
    launch.run_pip("install facefinder==0.1.4", "requirements for FaceFinder extension")
