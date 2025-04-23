#!/usr/bin/env python3

import os
import time
import uuid
import json
import argparse
import statistics
import boto3
import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class S3PerformanceTester:
    def __init__(self, bucket_prefix, region='us-east-1'):
        """Initialize the S3 performance tester with the given bucket prefix and region."""
        self.s3_client = boto3.client('s3', region_name=region)
        self.region = region
        self.bucket_prefix = bucket_prefix
        self.storage_classes = ['STANDARD', 'INTELLIGENT_TIERING', 'GLACIER_IR']
        self.buckets = {
            storage_class: f"{bucket_prefix}-{storage_class.lower().replace('_', '-')}"
            for storage_class in self.storage_classes
        }
        self.results = []
        
    def ensure_buckets_exist(self):
        """Make sure all required buckets exist, creating them if necessary."""
        existing_buckets = [bucket['Name'] for bucket in self.s3_client.list_buckets()['Buckets']]
        
        for storage_class, bucket_name in self.buckets.items():
            if bucket_name not in existing_buckets:
                print(f"Creating bucket: {bucket_name}")
                
                # Create bucket with location constraint if not in us-east-1
                if self.region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                
                # Enable CloudWatch metrics for the bucket
                self.s3_client.put_bucket_metrics_configuration(
                    Bucket=bucket_name,
                    Id='EntireBucket',
                    MetricsConfiguration={'Id': 'EntireBucket'}
                )
                
                print(f"Created bucket: {bucket_name}")
            else:
                print(f"Bucket already exists: {bucket_name}")
    
    def create_test_file(self, size_mb):
        """Create a test file of the specified size in MB."""
        filename = f"testfile_{size_mb}mb_{uuid.uuid4()}.bin"
        
        print(f"Creating test file: {filename} ({size_mb} MB)")
        
        # Create a file with random data
        with open(filename, 'wb') as f:
            f.write(os.urandom(size_mb * 1024 * 1024))
            
        return filename
    
    def measure_ttfb(self, bucket, key):
        """Measure Time To First Byte (TTFB) for an object."""
        # Generate a presigned URL to access the object
        presigned_url = self.s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=60
        )
        
        # Measure TTFB using streaming response
        start_time = time.time()
        response = requests.get(presigned_url, stream=True)
        
        # Get time to first byte
        ttfb = None
        for chunk in response.iter_content(chunk_size=1):
            if chunk:
                ttfb = time.time() - start_time
                break
                
        # Consume and close the rest of the response
        for _ in response.iter_content(chunk_size=8192):
            pass
        
        return ttfb * 1000  # Convert to milliseconds
    
    def test_single_put(self, file_path, storage_class, iterations=5):
        """Test single PUT upload and download operations."""
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        bucket = self.buckets[storage_class]
        
        print(f"Testing single PUT for {file_path} in {storage_class} storage class")
        
        upload_times = []
        download_times = []
        ttfb_values = []
        
        for i in range(iterations):
            key = f"single-put-{os.path.basename(file_path)}-{i}"
            
            # Measure upload time
            print(f"  Iteration {i+1}/{iterations} - Uploading...")
            start_time = time.time()
            self.s3_client.upload_file(
                file_path, 
                bucket, 
                key,
                ExtraArgs={'StorageClass': storage_class}
            )
            upload_time = time.time() - start_time
            upload_times.append(upload_time)
            
            # Measure TTFB
            print(f"  Iteration {i+1}/{iterations} - Measuring TTFB...")
            ttfb = self.measure_ttfb(bucket, key)
            ttfb_values.append(ttfb)
            
            # Measure download time
            print(f"  Iteration {i+1}/{iterations} - Downloading...")
            download_path = f"downloaded_{os.path.basename(file_path)}_{i}"
            start_time = time.time()
            self.s3_client.download_file(bucket, key, download_path)
            download_time = time.time() - start_time
            download_times.append(download_time)
            
            # Verify file integrity (size check)
            downloaded_size = os.path.getsize(download_path)
            original_size = os.path.getsize(file_path)
            if downloaded_size != original_size:
                print(f"WARNING: Size mismatch - Original: {original_size}, Downloaded: {downloaded_size}")
            
            # Clean up
            os.remove(download_path)
            self.s3_client.delete_object(Bucket=bucket, Key=key)
        
        # Calculate statistics
        upload_mean = statistics.mean(upload_times)
        upload_stdev = statistics.stdev(upload_times) if len(upload_times) > 1 else 0
        
        download_mean = statistics.mean(download_times)
        download_stdev = statistics.stdev(download_times) if len(download_times) > 1 else 0
        
        ttfb_mean = statistics.mean(ttfb_values)
        ttfb_stdev = statistics.stdev(ttfb_values) if len(ttfb_values) > 1 else 0
        
        # Calculate throughput
        upload_throughput = file_size_mb / upload_mean if upload_mean > 0 else 0
        download_throughput = file_size_mb / download_mean if download_mean > 0 else 0
        
        # Calculate confidence intervals (95%)
        confidence_factor = 1.96  # 95% confidence interval
        
        upload_ci = confidence_factor * upload_stdev / (len(upload_times) ** 0.5) if len(upload_times) > 1 else 0
        download_ci = confidence_factor * download_stdev / (len(download_times) ** 0.5) if len(download_times) > 1 else 0
        ttfb_ci = confidence_factor * ttfb_stdev / (len(ttfb_values) ** 0.5) if len(ttfb_values) > 1 else 0
        
        # Add to results
        result = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'single_put',
            'file_size_mb': file_size_mb,
            'storage_class': storage_class,
            'iterations': iterations,
            'upload_time_mean_s': upload_mean,
            'upload_time_stdev_s': upload_stdev,
            'upload_time_ci95_s': upload_ci,
            'download_time_mean_s': download_mean,
            'download_time_stdev_s': download_stdev,
            'download_time_ci95_s': download_ci,
            'ttfb_mean_ms': ttfb_mean,
            'ttfb_stdev_ms': ttfb_stdev,
            'ttfb_ci95_ms': ttfb_ci,
            'upload_throughput_mbps': upload_throughput,
            'download_throughput_mbps': download_throughput,
            'upload_vs_download_ratio': download_throughput / upload_throughput if upload_throughput > 0 else 0
        }
        
        self.results.append(result)
        print(f"Single PUT test completed for {file_path} in {storage_class}")
        return result
    
    def test_multipart_upload(self, file_path, storage_class, part_size_mb=5, iterations=5):
        """Test multipart upload and download operations."""
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        bucket = self.buckets[storage_class]
        
        # Skip multipart for small files
        if file_size_mb < 50:  # Only do multipart for files >= 50MB
            print(f"Skipping multipart test for {file_path} - file too small (<50MB)")
            return None
        
        print(f"Testing multipart upload for {file_path} in {storage_class} storage class")
        
        upload_times = []
        download_times = []
        ttfb_values = []
        
        for i in range(iterations):
            key = f"multipart-{os.path.basename(file_path)}-{i}"
            
            # Measure multipart upload time
            print(f"  Iteration {i+1}/{iterations} - Uploading multipart...")
            start_time = time.time()
            
            # Initiate multipart upload
            mpu = self.s3_client.create_multipart_upload(
                Bucket=bucket,
                Key=key,
                StorageClass=storage_class
            )
            upload_id = mpu['UploadId']
            
            # Process file in parts
            part_size = part_size_mb * 1024 * 1024  # part size in bytes
            file_size = os.path.getsize(file_path)
            part_count = (file_size + part_size - 1) // part_size
            
            parts = []
            with open(file_path, 'rb') as f:
                for part_num in range(1, part_count + 1):
                    data = f.read(part_size)
                    if not data:
                        break
                    
                    # Upload part
                    response = self.s3_client.upload_part(
                        Body=data,
                        Bucket=bucket,
                        Key=key,
                        PartNumber=part_num,
                        UploadId=upload_id
                    )
                    
                    parts.append({
                        'PartNumber': part_num,
                        'ETag': response['ETag']
                    })
            
            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
            upload_time = time.time() - start_time
            upload_times.append(upload_time)
            
            # Measure TTFB
            print(f"  Iteration {i+1}/{iterations} - Measuring TTFB...")
            ttfb = self.measure_ttfb(bucket, key)
            ttfb_values.append(ttfb)
            
            # Measure download time
            print(f"  Iteration {i+1}/{iterations} - Downloading...")
            download_path = f"downloaded_mp_{os.path.basename(file_path)}_{i}"
            start_time = time.time()
            self.s3_client.download_file(bucket, key, download_path)
            download_time = time.time() - start_time
            download_times.append(download_time)
            
            # Verify file integrity
            downloaded_size = os.path.getsize(download_path)
            original_size = os.path.getsize(file_path)
            if downloaded_size != original_size:
                print(f"WARNING: Size mismatch - Original: {original_size}, Downloaded: {downloaded_size}")
            
            # Clean up
            os.remove(download_path)
            self.s3_client.delete_object(Bucket=bucket, Key=key)
        
        # Calculate statistics
        upload_mean = statistics.mean(upload_times)
        upload_stdev = statistics.stdev(upload_times) if len(upload_times) > 1 else 0
        
        download_mean = statistics.mean(download_times)
        download_stdev = statistics.stdev(download_times) if len(download_times) > 1 else 0
        
        ttfb_mean = statistics.mean(ttfb_values)
        ttfb_stdev = statistics.stdev(ttfb_values) if len(ttfb_values) > 1 else 0
        
        # Calculate throughput
        upload_throughput = file_size_mb / upload_mean if upload_mean > 0 else 0
        download_throughput = file_size_mb / download_mean if download_mean > 0 else 0
        
        # Calculate confidence intervals (95%)
        confidence_factor = 1.96  # 95% confidence interval
        
        upload_ci = confidence_factor * upload_stdev / (len(upload_times) ** 0.5) if len(upload_times) > 1 else 0
        download_ci = confidence_factor * download_stdev / (len(download_times) ** 0.5) if len(download_times) > 1 else 0
        ttfb_ci = confidence_factor * ttfb_stdev / (len(ttfb_values) ** 0.5) if len(ttfb_values) > 1 else 0
        
        # Add to results
        result = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'multipart',
            'file_size_mb': file_size_mb,
            'storage_class': storage_class,
            'part_size_mb': part_size_mb,
            'iterations': iterations,
            'upload_time_mean_s': upload_mean,
            'upload_time_stdev_s': upload_stdev,
            'upload_time_ci95_s': upload_ci,
            'download_time_mean_s': download_mean,
            'download_time_stdev_s': download_stdev,
            'download_time_ci95_s': download_ci,
            'ttfb_mean_ms': ttfb_mean,
            'ttfb_stdev_ms': ttfb_stdev,
            'ttfb_ci95_ms': ttfb_ci,
            'upload_throughput_mbps': upload_throughput,
            'download_throughput_mbps': download_throughput,
            'upload_vs_download_ratio': download_throughput / upload_throughput if upload_throughput > 0 else 0
        }
        
        self.results.append(result)
        print(f"Multipart upload test completed for {file_path} in {storage_class}")
        return result
    
    def calculate_cost(self, operation_results):
        """Calculate theoretical cost for operations."""
        # Current S3 pricing (as of April 2025)
        prices = {
            'STANDARD': {
                'storage_gb_month': 0.023,
                'put_1000': 0.005,
                'get_1000': 0.0004,
                'data_transfer_gb': 0.09
            },
            'INTELLIGENT_TIERING': {
                'storage_gb_month': 0.023,
                'put_1000': 0.005,
                'get_1000': 0.0004,
                'data_transfer_gb': 0.09,
                'monitoring_fee_per_object': 0.0025 / 1000  # Per object per month
            },
            'GLACIER_IR': {
                'storage_gb_month': 0.004,
                'put_1000': 0.03,
                'get_1000': 0.01,
                'data_transfer_gb': 0.09
            }
        }
        
        # Clone the result and add cost metrics
        result = operation_results.copy()
        storage_class = result['storage_class']
        file_size_gb = result['file_size_mb'] / 1024
        price = prices[storage_class]
        
        # Calculate storage cost (daily)
        storage_cost = file_size_gb * price['storage_gb_month'] / 30
        
        # Calculate operation costs
        put_cost = price['put_1000'] / 1000
        get_cost = price['get_1000'] / 1000
        
        # Calculate transfer costs
        transfer_cost = file_size_gb * price['data_transfer_gb']
        
        # Calculate monitoring cost for Intelligent Tiering
        monitoring_cost = 0
        if storage_class == 'INTELLIGENT_TIERING':
            monitoring_cost = price['monitoring_fee_per_object']
        
        # Total cost
        total_cost = storage_cost + put_cost + get_cost + transfer_cost + monitoring_cost
        
        # Calculate cost efficiency (throughput per cost)
        cost_efficiency = result['upload_throughput_mbps'] / total_cost if total_cost > 0 else 0
        
        # Add cost metrics to result
        result['storage_cost_usd'] = storage_cost
        result['operation_cost_put_usd'] = put_cost
        result['operation_cost_get_usd'] = get_cost
        result['transfer_cost_usd'] = transfer_cost
        result['monitoring_cost_usd'] = monitoring_cost
        result['total_cost_usd'] = total_cost
        result['cost_per_gb_usd'] = total_cost / file_size_gb if file_size_gb > 0 else 0
        result['cost_efficiency_mbps_per_cent'] = cost_efficiency * 100  # MB/s per $0.01
        
        return result
    
    def run_tests(self, file_sizes, iterations=5):
        """Run all tests for the specified file sizes."""
        all_results = []
        
        # Make sure buckets exist
        self.ensure_buckets_exist()
        
        # Run tests for each file size
        for size_mb in file_sizes:
            print(f"\nTesting with {size_mb}MB files")
            
            # Create test file
            file_path = self.create_test_file(size_mb)
            
            # Test each storage class
            for storage_class in self.storage_classes:
                print(f"\nTesting storage class: {storage_class}")
                
                # Test single PUT
                single_put_result = self.test_single_put(file_path, storage_class, iterations)
                single_put_with_cost = self.calculate_cost(single_put_result)
                all_results.append(single_put_with_cost)
                
                # Test multipart upload (for larger files)
                if size_mb >= 50:
                    multipart_result = self.test_multipart_upload(file_path, storage_class, iterations=iterations)
                    if multipart_result:
                        multipart_with_cost = self.calculate_cost(multipart_result)
                        all_results.append(multipart_with_cost)
            
            # Clean up test file
            os.remove(file_path)
        
        # Save all results
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"s3_performance_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        print(f"\nAll tests completed. Results saved to {results_file}")
        return results_df
    
    def compare_upload_methods(self, results_df):
        """Compare single PUT vs. multipart upload performance."""
        # Filter results for files tested with both methods
        large_files = results_df[results_df['file_size_mb'] >= 50].copy()
        
        # Group by storage class and file size
        groups = large_files.groupby(['storage_class', 'file_size_mb'])
        
        comparisons = []
        
        for (storage_class, file_size), group in groups:
            single_put = group[group['operation'] == 'single_put']
            multipart = group[group['operation'] == 'multipart']
            
            if len(single_put) > 0 and len(multipart) > 0:
                single_time = single_put.iloc[0]['upload_time_mean_s']
                multipart_time = multipart.iloc[0]['upload_time_mean_s']
                
                single_throughput = single_put.iloc[0]['upload_throughput_mbps']
                multipart_throughput = multipart.iloc[0]['upload_throughput_mbps']
                
                time_improvement = ((single_time - multipart_time) / single_time) * 100
                throughput_improvement = ((multipart_throughput - single_throughput) / single_throughput) * 100
                
                comparison = {
                    'storage_class': storage_class,
                    'file_size_mb': file_size,
                    'single_put_time_s': single_time,
                    'multipart_time_s': multipart_time,
                    'time_improvement_pct': time_improvement,
                    'single_put_throughput_mbps': single_throughput,
                    'multipart_throughput_mbps': multipart_throughput,
                    'throughput_improvement_pct': throughput_improvement
                }
                
                comparisons.append(comparison)
        
        # Save comparisons
        if comparisons:
            comparisons_df = pd.DataFrame(comparisons)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            comparisons_file = f"s3_upload_method_comparison_{timestamp}.csv"
            comparisons_df.to_csv(comparisons_file, index=False)
            print(f"Upload method comparisons saved to {comparisons_file}")
            return comparisons_df
        
        return None
    
def main():
    """Main function to run the S3 performance tests."""
    parser = argparse.ArgumentParser(description='Test AWS S3 performance across storage classes')
    parser.add_argument('--bucket-prefix', required=True, help='Prefix for test buckets')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--iterations', type=int, default=5, help='Number of test iterations')
    args = parser.parse_args()
    
    # Initialize tester
    tester = S3PerformanceTester(args.bucket_prefix, args.region)
    
    # Define file sizes to test (in MB)
    file_sizes = [1, 10, 100, 1000]  # 1MB, 10MB, 100MB, 1GB
    
    # Run tests
    results_df = tester.run_tests(file_sizes, args.iterations)
    
    # Compare upload methods
    tester.compare_upload_methods(results_df)

if __name__ == '__main__':
    main()