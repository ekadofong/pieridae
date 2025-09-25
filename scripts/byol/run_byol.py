#!/usr/bin/env python3
"""
BYOL Cluster Runner
Utility script for submitting and monitoring BYOL jobs on Tiger cluster
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class BYOLJobManager:
    """Manages BYOL job submission and monitoring on Tiger cluster"""

    def __init__(self, base_dir: Path = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd().parent.parent
        self.scripts_dir = Path(__file__).parent  # scripts/byol directory
        self.results_dir = self.base_dir / "byol_results"

        # Slurm script paths
        self.scripts = {
            'train': self.scripts_dir / 'submit_byol_train.slurm',
            'train_gpu': self.scripts_dir / 'submit_byol_train_gpu.slurm',
            'analyze': self.scripts_dir / 'submit_byol_analyze.slurm',
            'full': self.scripts_dir / 'submit_byol_full.slurm',
            'full_gpu': self.scripts_dir / 'submit_byol_full_gpu.slurm'
        }

        # Verify script files exist
        for mode, script_path in self.scripts.items():
            if not script_path.exists():
                raise FileNotFoundError(f"Slurm script not found: {script_path}")

    def submit_job(self, mode: str, **kwargs) -> str:
        """Submit a BYOL job to Slurm"""
        if mode not in self.scripts:
            raise ValueError(f"Invalid mode: {mode}. Choose from: {list(self.scripts.keys())}")

        script_path = self.scripts[mode]

        # Build sbatch command
        cmd = ['sbatch']

        # Add any additional sbatch options
        if 'account' in kwargs:
            cmd.extend(['--account', kwargs['account']])
        if 'partition' in kwargs:
            cmd.extend(['--partition', kwargs['partition']])
        if 'time' in kwargs:
            cmd.extend(['--time', kwargs['time']])
        if 'mem' in kwargs:
            cmd.extend(['--mem', kwargs['mem']])
        if 'cpus' in kwargs:
            cmd.extend(['--cpus-per-task', str(kwargs['cpus'])])
        if 'gres' in kwargs:
            cmd.extend(['--gres', kwargs['gres']])

        # Add script path
        cmd.append(str(script_path))

        # Submit job
        print(f"Submitting {mode} job...")
        print(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]
            print(f"Job submitted successfully. Job ID: {job_id}")

            # Save job info
            self._save_job_info(job_id, mode, kwargs)

            return job_id

        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e}")
            print(f"stderr: {e.stderr}")
            raise

    def _save_job_info(self, job_id: str, mode: str, kwargs: Dict):
        """Save job information for tracking"""
        job_info = {
            'job_id': job_id,
            'mode': mode,
            'submit_time': datetime.now().isoformat(),
            'status': 'submitted',
            'kwargs': kwargs
        }

        jobs_file = self.results_dir / 'jobs.json'
        self.results_dir.mkdir(exist_ok=True)

        # Load existing jobs
        jobs = []
        if jobs_file.exists():
            with open(jobs_file, 'r') as f:
                jobs = json.load(f)

        # Add new job
        jobs.append(job_info)

        # Save updated jobs
        with open(jobs_file, 'w') as f:
            json.dump(jobs, f, indent=2)

    def check_job_status(self, job_id: str = None) -> Dict:
        """Check status of BYOL jobs"""
        if job_id:
            cmd = ['squeue', '-j', job_id, '--format=%i,%T,%M,%N,%r']
        else:
            # Check all user jobs with byol in name
            cmd = ['squeue', '-u', os.environ.get('USER', ''), '--name=byol*', '--format=%i,%T,%M,%N,%r']

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')[1:]  # Skip header

            jobs = []
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 5:
                        jobs.append({
                            'job_id': parts[0],
                            'status': parts[1],
                            'time': parts[2],
                            'nodes': parts[3],
                            'reason': parts[4] if len(parts) > 4 else ''
                        })

            return jobs

        except subprocess.CalledProcessError as e:
            print(f"Error checking job status: {e}")
            return []

    def cancel_job(self, job_id: str):
        """Cancel a BYOL job"""
        cmd = ['scancel', job_id]

        try:
            subprocess.run(cmd, check=True)
            print(f"Job {job_id} cancelled successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error cancelling job {job_id}: {e}")

    def get_job_logs(self, job_id: str) -> str:
        """Get job output logs"""
        log_patterns = [
            f"slurm-{job_id}.out",
            f"slurm-{job_id}.err",
        ]

        logs = {}
        for pattern in log_patterns:
            log_file = self.base_dir / pattern
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs[pattern] = f.read()

        # Also check for logs in results directories
        if self.results_dir.exists():
            for result_dir in self.results_dir.iterdir():
                if result_dir.is_dir():
                    log_file = result_dir / f"slurm_job_{job_id}.log"
                    if log_file.exists():
                        with open(log_file, 'r') as f:
                            logs[f"job_log_{job_id}"] = f.read()

        return logs

    def monitor_job(self, job_id: str, interval: int = 30):
        """Monitor job progress with periodic updates"""
        print(f"Monitoring job {job_id}...")
        print("Press Ctrl+C to stop monitoring (job will continue running)")

        try:
            while True:
                jobs = self.check_job_status(job_id)

                if not jobs:
                    print(f"Job {job_id} completed or not found")
                    break

                job = jobs[0]
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Job {job_id}: {job['status']} (Time: {job['time']}, Node: {job['nodes']})")

                if job['status'] in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    print(f"Job {job_id} finished with status: {job['status']}")
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\nStopped monitoring job {job_id} (job continues running)")

    def list_results(self):
        """List available results"""
        if not self.results_dir.exists():
            print("No results directory found")
            return

        print(f"Results in {self.results_dir}:")
        for item in sorted(self.results_dir.iterdir()):
            if item.is_dir():
                # Check for key files
                files = list(item.glob('*'))
                key_files = []
                for f in files:
                    if f.name in ['byol_final_model.pt', 'embeddings.npy', 'analysis_summary.json']:
                        key_files.append(f.name)

                print(f"  {item.name}/ - {len(files)} files ({', '.join(key_files)})")

    def cleanup_old_results(self, days_old: int = 7):
        """Remove old result directories"""
        if not self.results_dir.exists():
            return

        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        removed = 0

        for item in self.results_dir.iterdir():
            if item.is_dir() and item.stat().st_mtime < cutoff_time:
                import shutil
                shutil.rmtree(item)
                removed += 1
                print(f"Removed old result directory: {item.name}")

        print(f"Cleaned up {removed} old result directories")


def main():
    parser = argparse.ArgumentParser(description='BYOL Cluster Job Manager')
    parser.add_argument('action', choices=['submit', 'status', 'monitor', 'cancel', 'logs', 'list', 'cleanup'],
                       help='Action to perform')
    parser.add_argument('--mode', choices=['train', 'train_gpu', 'analyze', 'full', 'full_gpu'], default='full',
                       help='BYOL analysis mode')
    parser.add_argument('--job-id', help='Job ID for status/monitor/cancel/logs actions')
    parser.add_argument('--account', help='Slurm account to use')
    parser.add_argument('--partition', help='Slurm partition')
    parser.add_argument('--time', help='Job time limit')
    parser.add_argument('--mem', help='Memory requirement')
    parser.add_argument('--cpus', type=int, help='Number of CPUs')
    parser.add_argument('--gres', help='Generic resources (e.g., gpu:1)')
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval in seconds')
    parser.add_argument('--days', type=int, default=7, help='Days old for cleanup')

    args = parser.parse_args()

    try:
        manager = BYOLJobManager()

        if args.action == 'submit':
            kwargs = {}
            for key in ['account', 'partition', 'time', 'mem', 'cpus', 'gres']:
                if getattr(args, key):
                    kwargs[key] = getattr(args, key)

            job_id = manager.submit_job(args.mode, **kwargs)
            print(f"To monitor: python run_byol.py monitor --job-id {job_id}")

        elif args.action == 'status':
            jobs = manager.check_job_status(args.job_id)
            if jobs:
                print("Job Status:")
                for job in jobs:
                    print(f"  ID: {job['job_id']}, Status: {job['status']}, "
                          f"Time: {job['time']}, Node: {job['nodes']}")
            else:
                print("No running BYOL jobs found" if not args.job_id else f"Job {args.job_id} not found")

        elif args.action == 'monitor':
            if not args.job_id:
                print("--job-id required for monitor action")
                sys.exit(1)
            manager.monitor_job(args.job_id, args.interval)

        elif args.action == 'cancel':
            if not args.job_id:
                print("--job-id required for cancel action")
                sys.exit(1)
            manager.cancel_job(args.job_id)

        elif args.action == 'logs':
            if not args.job_id:
                print("--job-id required for logs action")
                sys.exit(1)
            logs = manager.get_job_logs(args.job_id)
            for log_name, content in logs.items():
                print(f"=== {log_name} ===")
                print(content)
                print()

        elif args.action == 'list':
            manager.list_results()

        elif args.action == 'cleanup':
            manager.cleanup_old_results(args.days)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()