@echo off
rem Trail-camera VLM analysis with the CPU (no GPU needed) settings in profiles\cpu.json. Pass a folder or options, e.g.
rem   run_vlm_cpu.cmd "D:\Trail Photos\Camera 1" --limit 20
rem Hand over to run_vlm.cmd (without "call", which would expand % and ^ in the arguments again).
"%~dp0run_vlm.cmd" --profile cpu %*
