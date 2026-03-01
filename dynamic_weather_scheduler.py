"""
Dynamic Weather Period Selector for RL HVAC Control

This script automatically changes the simulation start time for each episode
within a specified time range, allowing the RL agent to experience
different weather conditions throughout the year.
"""

import os
import sys
import random
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import yaml


class DynamicWeatherScheduler:
    """Schedules different weather periods for each RL episode."""
    
    def __init__(self, config_path="config/hvac_config.yaml"):
        self.config_path = config_path
        self.project_root = Path(__file__).parent.parent
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file."""
        config_file = self.project_root / self.config_path
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_episode_schedule(self, num_episodes, time_range):
        """
        Generate episode schedule with random start times within specified range.
        
        Args:
            num_episodes: Number of episodes to run
            time_range: Dict with 'start_month', 'start_day', 'end_month', 'end_day'
        
        Returns:
            List of dicts with episode info
        """
        episodes = []
        
        # Convert to datetime objects for easier calculation
        start_date = datetime(2023, time_range['start_month'], time_range['start_day'])
        end_date = datetime(2023, time_range['end_month'], time_range['end_day'])
        
        # Calculate total days in range
        total_days = (end_date - start_date).days + 1
        
        print(f"Time range: {time_range['start_month']}/{time_range['start_day']} to {time_range['end_month']}/{time_range['end_day']}")
        print(f"Total days available: {total_days}")
        print(f"Generating {num_episodes} episodes...")
        
        for episode_num in range(num_episodes):
            # Random start day within range
            random_days_offset = random.randint(0, total_days - 1)
            episode_start = start_date + timedelta(days=random_days_offset)
            
            # Calculate day of year for weather selection
            day_of_year = episode_start.timetuple().tm_yday
            
            episodes.append({
                'episode': episode_num + 1,
                'start_date': episode_start,
                'month': episode_start.month,
                'day': episode_start.day,
                'day_of_year': day_of_year,
                'season': self.get_season(episode_start.month)
            })
            
            print(f"  Episode {episode_num + 1}: {episode_start.strftime('%B %d')} (Day {day_of_year})")
        
        return episodes
    
    def get_season(self, month):
        """Get season name for month."""
        seasons = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall"
        }
        return seasons.get(month, "Unknown")
    
    def create_episode_idf(self, original_idf, episode_info, output_dir):
        """
        Create a modified IDF for a specific episode with custom start date.
        
        Args:
            original_idf: Path to original IDF file
            episode_info: Dict with episode information
            output_dir: Directory to save modified IDF
        
        Returns:
            Path to modified IDF file
        """
        # Read original IDF
        with open(original_idf, 'r') as f:
            idf_content = f.read()
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(original_idf))[0]
        episode_idf_path = os.path.join(output_dir, 
            f"{base_name}_episode_{episode_info['episode']:03d}_{episode_info['month']:02d}{episode_info['day']:02d}.idf")
        
        # Find and replace RunPeriod section
        lines = idf_content.split('\n')
        new_lines = []
        in_runperiod = False
        runperiod_found = False
        
        for line in lines:
            if line.strip().startswith('RunPeriod,'):
                in_runperiod = True
                runperiod_found = True
                # Replace with episode-specific run period
                new_lines.extend([
                    '  RunPeriod,',
                    f'    EPISODE_{episode_info["episode"]:03d},    !- Name',
                    f'    {episode_info["month"]},            !- Begin Month',
                    f'    {episode_info["day"]},              !- Begin Day of Month',
                    f'    ,                        !- Begin Year',
                    f'    {episode_info["month"]},              !- End Month',
                    f'    {episode_info["day"]},                !- End Day of Month',
                    f'    ,                        !- End Year',
                    '    Sunday,                  !- Day of Week for Start Day',
                    '    No,                      !- Use Weather File Holidays and Special Days',
                    '    No,                      !- Use Weather File Daylight Saving Period',
                    '    No,                      !- Apply Weekend Holiday Rule',
                    '    Yes,                     !- Use Weather File Rain Indicators',
                    '    Yes;                     !- Use Weather File Snow Indicators',
                    ''
                ])
            elif in_runperiod and line.strip().endswith(';'):
                # End of RunPeriod block
                in_runperiod = False
            elif not in_runperiod:
                # Keep all other lines (including SizingPeriod objects)
                new_lines.append(line)
        
        if not runperiod_found:
            raise ValueError("No RunPeriod found in IDF file")
        
        # Write modified IDF
        with open(episode_idf_path, 'w') as f:
            f.write('\n'.join(new_lines))
        
        return episode_idf_path
    
    def run_dynamic_episodes(self, num_episodes=10, time_range=None):
        """
        Run multiple episodes with dynamically changing weather periods.
        
        Args:
            num_episodes: Number of episodes to run
            time_range: Dict with time range, defaults to full year
        """
        if time_range is None:
            # Default to full year range
            time_range = {
                'start_month': 1, 'start_day': 1,
                'end_month': 12, 'end_day': 31
            }
        
        print("=" * 70)
        print("DYNAMIC WEATHER EPISODE SCHEDULER")
        print("=" * 70)
        
        # Generate episode schedule
        episodes = self.get_episode_schedule(num_episodes, time_range)
        
        # Get paths
        original_idf = self.project_root / "energyplus/control_models/MediumOffice_IAQ.idf"
        weather_file = self.project_root / "weather/USA_CO_Denver.Intl.AP.724650_TMY3.epw"
        output_dir = self.project_root / "outputs/dynamic_episodes"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create IDF for each episode
        episode_idfs = []
        for episode_info in episodes:
            episode_idf = self.create_episode_idf(original_idf, episode_info, output_dir)
            episode_idfs.append(episode_idf)
        
        print(f"\nCreated {len(episode_idfs)} episode IDF files in {output_dir}")
        
        # Run episodes
        results = []
        for i, episode_info in enumerate(episodes):
            print(f"\n{'='*60}")
            print(f"RUNNING EPISODE {episode_info['episode']}/{num_episodes}")
            print(f"Weather: {episode_info['season']} {episode_info['month']}/{episode_info['day']}")
            print(f"Day of year: {episode_info['day_of_year']}")
            print(f"{'='*60}")
            
            # Run single episode
            result = self.run_single_episode(
                episode_idfs[i], 
                weather_file, 
                output_dir,
                episode_info
            )
            results.append(result)
        
        # Print summary
        self.print_episode_summary(results)
        
        return results
    
    def run_single_episode(self, episode_idf, weather_file, output_dir, episode_info):
        """
        Run a single episode with specified weather period.
        
        Returns:
            Dict with episode results
        """
        try:
            # Command to run RL HVAC control
            cmd = [
                sys.executable, 
                "tests/rl_hvac_control.py",
                "--idf", episode_idf,
                "--epw", str(weather_file),
                "--output", output_dir,
                "--config", "config/hvac_config.yaml",
                "--episodes", "1"  # Run 1 episode per IDF
            ]
            
            # Run the episode
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            return {
                'episode': episode_info['episode'],
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'season': episode_info['season'],
                'month': episode_info['month'],
                'day': episode_info['day']
            }
            
        except Exception as e:
            print(f"Error running episode {episode_info['episode']}: {e}")
            return {
                'episode': episode_info['episode'],
                'success': False,
                'error': str(e),
                'season': episode_info['season']
            }
    
    def print_episode_summary(self, results):
        """Print summary of all episodes."""
        print(f"\n{'='*70}")
        print("EPISODE SUMMARY")
        print(f"{'='*70}")
        
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        print(f"Total episodes: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success rate: {successful/total*100:.1f}%")
        
        # Group by season
        seasons = {}
        for result in results:
            season = result['season']
            if season not in seasons:
                seasons[season] = {'total': 0, 'success': 0}
            seasons[season]['total'] += 1
            if result['success']:
                seasons[season]['success'] += 1
        
        print(f"\nSeason breakdown:")
        for season, counts in seasons.items():
            success_rate = counts['success']/counts['total']*100 if counts['total'] > 0 else 0
            print(f"  {season}: {counts['success']}/{counts['total']} ({success_rate:.1f}%)")
        
        print(f"{'='*70}")


def main():
    """Main function to run dynamic weather episodes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RL HVAC with dynamic weather periods")
    parser.add_argument('--episodes', type=int, default=10, 
                       help='Number of episodes to run')
    parser.add_argument('--start-month', type=int, default=1,
                       help='Start month (1-12)')
    parser.add_argument('--start-day', type=int, default=1,
                       help='Start day (1-31)')
    parser.add_argument('--end-month', type=int, default=12,
                       help='End month (1-12)')
    parser.add_argument('--end-day', type=int, default=31,
                       help='End day (1-31)')
    parser.add_argument('--config', type=str, default='config/hvac_config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Create scheduler
    scheduler = DynamicWeatherScheduler(args.config)
    
    # Define time range
    time_range = {
        'start_month': args.start_month,
        'start_day': args.start_day,
        'end_month': args.end_month,
        'end_day': args.end_day
    }
    
    # Run dynamic episodes
    scheduler.run_dynamic_episodes(args.episodes, time_range)


if __name__ == "__main__":
    main()
