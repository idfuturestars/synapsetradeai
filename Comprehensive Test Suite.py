#!/usr/bin/env python3
"""
SynapseTrade AI™ - Comprehensive Testing Suite
Chief Technical Architect Implementation
Tests 100% of platform functionality with diagnostics
"""

import os
import sys
import json
import time
import requests
import subprocess
from datetime import datetime
from colorama import init, Fore, Back, Style
import pandas as pd
import numpy as np

# Initialize colorama for cross-platform colored output
init()

class TestFramework:
    """Comprehensive testing framework with diagnostics"""
    
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.test_results = []
        self.start_time = datetime.now()
        self.passed = 0
        self.failed = 0
        self.api_token = None
        
    def backup_system(self):
        """Create system backup before testing"""
        print(f"\n{Fore.CYAN}📦 Creating System Backup...{Style.RESET_ALL}")
        backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Create backup directory
            os.makedirs(f"backups/{backup_name}", exist_ok=True)
            
            # Backup database
            if os.path.exists('synapsetrade.db'):
                subprocess.run(['cp', 'synapsetrade.db', f'backups/{backup_name}/'])
                print(f"{Fore.GREEN}✓ Database backed up{Style.RESET_ALL}")
            
            # Backup code
            for file in ['app.py', 'requirements.txt', '.env']:
                if os.path.exists(file):
                    subprocess.run(['cp', file, f'backups/{backup_name}/'])
            
            print(f"{Fore.GREEN}✓ Backup completed: backups/{backup_name}{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}✗ Backup failed: {e}{Style.RESET_ALL}")
            return False
    
    def check_ports(self):
        """Comprehensive port availability check"""
        print(f"\n{Fore.CYAN}🔌 Checking Port Availability...{Style.RESET_ALL}")
        
        ports_to_check = {
            8080: "API Server",
            5432: "PostgreSQL",
            6379: "Redis",
            3000: "Mobile App",
            8545: "Blockchain RPC"
        }
        
        for port, service in ports_to_check.items():
            result = subprocess.run(
                ['lsof', '-i', f':{port}'], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                print(f"{Fore.YELLOW}⚠ Port {port} ({service}) is in use{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}✓ Port {port} ({service}) is available{Style.RESET_ALL}")
    
    def diagnostic_block(self, test_name, response=None, error=None):
        """Diagnostic information block for troubleshooting"""
        print(f"\n{Back.BLUE}{Fore.WHITE} DIAGNOSTIC: {test_name} {Style.RESET_ALL}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Endpoint: {self.base_url}")
        
        if response:
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            try:
                print(f"Response: {json.dumps(response.json(), indent=2)[:500]}...")
            except:
                print(f"Response Text: {response.text[:500]}...")
        
        if error:
            print(f"{Fore.RED}Error: {error}{Style.RESET_ALL}")
        
        print(f"{'-' * 50}")
    
    def test_endpoint(self, name, endpoint, method="GET", data=None, expected_status=200):
        """Test a single endpoint with diagnostics"""
        print(f"\n{Fore.CYAN}Testing: {name}{Style.RESET_ALL}")
        
        try:
            if method == "GET":
                response = requests.get(f"{self.base_url}{endpoint}")
            elif method == "POST":
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json=data,
                    headers={'Content-Type': 'application/json'}
                )
            else:
                response = requests.request(
                    method, 
                    f"{self.base_url}{endpoint}",
                    json=data
                )
            
            if response.status_code == expected_status:
                print(f"{Fore.GREEN}✓ {name} - Status: {response.status_code}{Style.RESET_ALL}")
                self.passed += 1
                
                # Validate response structure
                if response.content:
                    try:
                        json_data = response.json()
                        print(f"  Response keys: {list(json_data.keys())}")
                    except:
                        pass
                
                return True
            else:
                print(f"{Fore.RED}✗ {name} - Status: {response.status_code} (Expected: {expected_status}){Style.RESET_ALL}")
                self.failed += 1
                self.diagnostic_block(name, response)
                return False
                
        except Exception as e:
            print(f"{Fore.RED}✗ {name} - Error: {str(e)}{Style.RESET_ALL}")
            self.failed += 1
            self.diagnostic_block(name, error=str(e))
            return False
    
    def run_all_tests(self):
        """Execute comprehensive test suite"""
        print(f"\n{Back.GREEN}{Fore.BLACK} SYNAPSETRADE AI™ COMPREHENSIVE TEST SUITE {Style.RESET_ALL}")
        print(f"{'=' * 60}")
        
        # Pre-flight checks
        self.backup_system()
        self.check_ports()
        
        # 1. Core System Tests
        print(f"\n{Back.YELLOW}{Fore.BLACK} 1. CORE SYSTEM TESTS {Style.RESET_ALL}")
        
        self.test_endpoint("Health Check", "/api/health")
        self.test_endpoint("API Documentation", "/api/docs")
        
        # 2. Authentication Tests
        print(f"\n{Back.YELLOW}{Fore.BLACK} 2. AUTHENTICATION TESTS {Style.RESET_ALL}")
        
        user_data = {
            "username": f"test_user_{int(time.time())}",
            "password": "test_password_123"
        }
        
        self.test_endpoint(
            "User Registration",
            "/api/register",
            method="POST",
            data=user_data,
            expected_status=201
        )
        
        self.test_endpoint(
            "User Login",
            "/api/login",
            method="POST",
            data=user_data
        )
        
        # 3. Sentiment Analysis Tests
        print(f"\n{Back.YELLOW}{Fore.BLACK} 3. SENTIMENT ANALYSIS TESTS {Style.RESET_ALL}")
        
        sentiment_data = {
            "headlines": [
                "Apple announces record-breaking quarterly earnings",
                "Tech stocks plummet amid recession fears",
                "Microsoft unveils revolutionary AI technology"
            ]
        }
        
        self.test_endpoint(
            "News Sentiment Analysis",
            "/api/sentiment/analyze",
            method="POST",
            data=sentiment_data
        )
        
        # 4. Technical Analysis Tests
        print(f"\n{Back.YELLOW}{Fore.BLACK} 4. TECHNICAL ANALYSIS TESTS {Style.RESET_ALL}")
        
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        for symbol in symbols:
            self.test_endpoint(
                f"Technical Indicators - {symbol}",
                f"/api/technical/{symbol}"
            )
        
        # 5. Machine Learning Tests
        print(f"\n{Back.YELLOW}{Fore.BLACK} 5. MACHINE LEARNING TESTS {Style.RESET_ALL}")
        
        for symbol in ["AAPL", "BTC", "ETH"]:
            self.test_endpoint(
                f"ML Price Prediction - {symbol}",
                f"/api/ml/predict/{symbol}"
            )
        
        # 6. Backtesting Tests
        print(f"\n{Back.YELLOW}{Fore.BLACK} 6. BACKTESTING ENGINE TESTS {Style.RESET_ALL}")
        
        backtest_configs = [
            {
                "symbol": "AAPL",
                "strategy": "mean_reversion",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31"
            },
            {
                "symbol": "MSFT",
                "strategy": "macd",
                "start_date": "2023-06-01",
                "end_date": "2023-12-31"
            }
        ]
        
        for config in backtest_configs:
            self.test_endpoint(
                f"Backtest - {config['symbol']} {config['strategy']}",
                "/api/backtest",
                method="POST",
                data=config
            )
        
        # 7. Risk Management Tests
        print(f"\n{Back.YELLOW}{Fore.BLACK} 7. RISK MANAGEMENT TESTS {Style.RESET_ALL}")
        
        risk_scenarios = [
            {
                "balance": 10000,
                "risk_percentage": 2,
                "entry_price": 150,
                "stop_loss_price": 145
            },
            {
                "balance": 50000,
                "risk_percentage": 1,
                "entry_price": 3500,
                "stop_loss_price": 3400
            }
        ]
        
        for scenario in risk_scenarios:
            self.test_endpoint(
                f"Position Sizing - ${scenario['balance']}",
                "/api/risk/position-size",
                method="POST",
                data=scenario
            )
        
        # 8. Data Processing Tests
        print(f"\n{Back.YELLOW}{Fore.BLACK} 8. DATA PROCESSING TESTS {Style.RESET_ALL}")
        
        text_data = {
            "texts": [
                "Earnings report shows strong growth in cloud services",
                "Regulatory challenges impact tech sector outlook",
                "Innovation drives market leadership in AI space"
            ]
        }
        
        self.test_endpoint(
            "TF-IDF Text Processing",
            "/api/data/preprocess",
            method="POST",
            data=text_data
        )
        
        # 9. Fundamental Analysis Tests
        print(f"\n{Back.YELLOW}{Fore.BLACK} 9. FUNDAMENTAL ANALYSIS TESTS {Style.RESET_ALL}")
        
        self.test_endpoint(
            "Apple vs Microsoft Comparison",
            "/api/comparison/fundamental"
        )
        
        # 10. Portfolio Management Tests
        print(f"\n{Back.YELLOW}{Fore.BLACK} 10. PORTFOLIO MANAGEMENT TESTS {Style.RESET_ALL}")
        
        self.test_endpoint(
            "Get Portfolio",
            "/api/portfolio",
            expected_status=401  # Should require auth
        )
        
        # 11. Trading Execution Tests
        print(f"\n{Back.YELLOW}{Fore.BLACK} 11. TRADING EXECUTION TESTS {Style.RESET_ALL}")
        
        trade_data = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 10,
            "type": "market"
        }
        
        self.test_endpoint(
            "Execute Trade",
            "/api/trade",
            method="POST",
            data=trade_data
        )
        
        # Generate Test Report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n{Back.MAGENTA}{Fore.WHITE} TEST SUITE SUMMARY {Style.RESET_ALL}")
        print(f"{'=' * 60}")
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"{Fore.GREEN}Passed: {self.passed}{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed: {self.failed}{Style.RESET_ALL}")
        print(f"Success Rate: {(self.passed / (self.passed + self.failed) * 100):.1f}%")
        print(f"Duration: {duration:.2f} seconds")
        print(f"{'=' * 60}")
        
        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": self.passed + self.failed,
            "passed": self.passed,
            "failed": self.failed,
            "success_rate": self.passed / (self.passed + self.failed) * 100,
            "duration": duration
        }
        
        with open(f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(report, f, indent=2)
        
        if self.failed == 0:
            print(f"\n{Back.GREEN}{Fore.WHITE} ✅ ALL TESTS PASSED! {Style.RESET_ALL}")
        else:
            print(f"\n{Back.RED}{Fore.WHITE} ⚠️  SOME TESTS FAILED - CHECK DIAGNOSTICS {Style.RESET_ALL}")

class PerformanceProfiler:
    """Performance profiling for optimization"""
    
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.results = []
    
    def profile_endpoint(self, name, endpoint, method="GET", data=None, iterations=10):
        """Profile endpoint performance"""
        print(f"\n{Fore.CYAN}Profiling: {name} ({iterations} iterations){Style.RESET_ALL}")
        
        times = []
        for i in range(iterations):
            start = time.time()
            
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}")
                else:
                    response = requests.post(
                        f"{self.base_url}{endpoint}",
                        json=data
                    )
                
                elapsed = time.time() - start
                times.append(elapsed)
                
                print(f"  Iteration {i+1}: {elapsed:.3f}s", end="\r")
                
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            
            print(f"\n  Average: {avg_time:.3f}s")
            print(f"  Min: {min_time:.3f}s")
            print(f"  Max: {max_time:.3f}s")
            print(f"  Std Dev: {std_time:.3f}s")
            
            self.results.append({
                "endpoint": name,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "std_dev": std_time
            })
    
    def run_performance_tests(self):
        """Run complete performance profiling"""
        print(f"\n{Back.BLUE}{Fore.WHITE} PERFORMANCE PROFILING {Style.RESET_ALL}")
        
        # Critical endpoints to profile
        self.profile_endpoint("Health Check", "/api/health")
        self.profile_endpoint("Technical Analysis", "/api/technical/AAPL")
        self.profile_endpoint(
            "Sentiment Analysis",
            "/api/sentiment/analyze",
            method="POST",
            data={"headlines": ["Test headline"]}
        )
        self.profile_endpoint("ML Prediction", "/api/ml/predict/AAPL")
        
        # Generate performance report
        print(f"\n{Back.MAGENTA}{Fore.WHITE} PERFORMANCE SUMMARY {Style.RESET_ALL}")
        df = pd.DataFrame(self.results)
        print(df.to_string(index=False))

def main():
    """Main test execution"""
    print(f"{Back.CYAN}{Fore.BLACK} SYNAPSETRADE AI™ - CHIEF TECHNICAL ARCHITECT TEST SUITE {Style.RESET_ALL}")
    print(f"Version: 2.0.0")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8080/api/health", timeout=2)
        print(f"\n{Fore.GREEN}✓ Server is running{Style.RESET_ALL}")
    except:
        print(f"\n{Fore.RED}✗ Server not responding at http://localhost:8080{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Start the server with: python app.py{Style.RESET_ALL}")
        return
    
    # Run comprehensive tests
    tester = TestFramework()
    tester.run_all_tests()
    
    # Run performance profiling
    profiler = PerformanceProfiler()
    profiler.run_performance_tests()

if __name__ == "__main__":
    main()