@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
call conda activate bem

echo ================================================================
echo DYNAMIC WEATHER EPISODE SCHEDULER
echo ================================================================
echo.
echo This script runs multiple RL episodes with random weather periods
echo Each episode gets a different start date within the specified range
echo.

python dynamic_weather_scheduler.py --episodes 10 --start-month 6 --start-day 1 --end-month 8 --end-day 31

echo.
echo ================================================================
echo Episodes completed! Check outputs/dynamic_episodes/ for results
echo ================================================================

pause
