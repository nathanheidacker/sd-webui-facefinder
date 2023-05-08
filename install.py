import launch

if not launch.is_installed("facefinder==0.1.5"):
    launch.run_pip("install facefinder==0.1.5", "requirements for FaceFinder extension")
