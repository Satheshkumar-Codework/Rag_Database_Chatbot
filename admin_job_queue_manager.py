# admin_job_queue_manager.py
"""
Enhanced Job Queue Manager for Admin Operations
Handles multiple PDF uploads without interrupting background processing
"""

import asyncio
import asyncpg
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import os
from pathlib import Path

class JobStatus(Enum):
    QUEUED = "QUEUED"           # Job is in queue waiting to start
    PENDING = "PENDING"         # Job is ready to be picked up by worker
    PROCESSING = "PROCESSING"   # Job is currently being processed
    COMPLETED = "COMPLETED"     # Job completed successfully
    FAILED = "FAILED"          # Job failed
    PAUSED = "PAUSED"          # Job is temporarily paused

class JobPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    ADMIN = 4                  # Highest priority for admin operations

class DocumentStatus(Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"

@dataclass
class AdminJob:
    id: str
    admin_id: str
    filename: str
    file_path: str
    job_type: str
    status: JobStatus
    priority: JobPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    estimated_duration: int = 300  # seconds
    dependencies: List[str] = field(default_factory=list)

@dataclass
class JobBatch:
    batch_id: str
    admin_id: str
    jobs: List[AdminJob]
    created_at: datetime
    total_jobs: int
    completed_jobs: int = 0
    failed_jobs: int = 0

class AdminJobQueueManager:
    def __init__(self, db_config: Dict, max_workers: int = 5, admin_workers: int = 3):
        self.db_config = db_config
        self.max_workers = max_workers
        self.admin_workers = admin_workers  # Dedicated workers for admin tasks
        
        # Thread pools
        self.general_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.admin_executor = ThreadPoolExecutor(max_workers=admin_workers)
        
        # Job queues
        self.admin_queue = queue.PriorityQueue()
        self.general_queue = queue.PriorityQueue()
        
        # Running jobs tracking
        self.running_jobs: Set[str] = set()
        self.admin_running_jobs: Set[str] = set()
        
        # Job handlers
        self.job_handlers: Dict[str, Callable] = {}
        
        # Batch tracking
        self.active_batches: Dict[str, JobBatch] = {}
        
        # Control flags
        self.running = False
        self.paused_admins: Set[str] = set()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Job status cache for quick access
        self.job_status_cache: Dict[str, AdminJob] = {}
        
        # Background tasks
        self.monitor_thread = None
        self.cleanup_thread = None
    
    async def get_db_connection(self):
        """Get database connection with connection pooling"""
        return await asyncpg.connect(**self.db_config)
    
    def register_job_handler(self, job_type: str, handler: Callable):
        """Register a handler function for a specific job type"""
        self.job_handlers[job_type] = handler
    
    async def create_job_tables(self):
        """Create enhanced job tables with admin support"""
        conn = await self.get_db_connection()
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS admin_pdf_jobs (
                    id VARCHAR(255) PRIMARY KEY,
                    admin_id VARCHAR(255) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    file_path VARCHAR(500) NOT NULL,
                    job_type VARCHAR(50) NOT NULL,
                    status VARCHAR(50) DEFAULT 'QUEUED',
                    priority INTEGER DEFAULT 2,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    estimated_duration INTEGER DEFAULT 300,
                    actual_duration INTEGER,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    error_message TEXT,
                    metadata JSONB DEFAULT '{}',
                    dependencies TEXT[] DEFAULT ARRAY[]::TEXT[],
                    batch_id VARCHAR(255)
                );
                
                CREATE TABLE IF NOT EXISTS admin_job_batches (
                    batch_id VARCHAR(255) PRIMARY KEY,
                    admin_id VARCHAR(255) NOT NULL,
                    total_jobs INTEGER NOT NULL,
                    completed_jobs INTEGER DEFAULT 0,
                    failed_jobs INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'PROCESSING'
                );
                
                CREATE TABLE IF NOT EXISTS admin_document_status (
                    id SERIAL PRIMARY KEY,
                    admin_id VARCHAR(255) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    embedding_status VARCHAR(50) DEFAULT 'QUEUED',
                    summarization_status VARCHAR(50) DEFAULT 'QUEUED',
                    overall_status VARCHAR(50) DEFAULT 'QUEUED',
                    embedding_job_id VARCHAR(255),
                    summarization_job_id VARCHAR(255),
                    batch_id VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_start_time TIMESTAMP,
                    processing_end_time TIMESTAMP,
                    UNIQUE(admin_id, filename, batch_id)
                );
                
                CREATE TABLE IF NOT EXISTS job_performance_metrics (
                    id SERIAL PRIMARY KEY,
                    job_type VARCHAR(50) NOT NULL,
                    avg_duration INTEGER,
                    success_rate DECIMAL(5,2),
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Indexes for better performance
                CREATE INDEX IF NOT EXISTS idx_admin_jobs_status_priority ON admin_pdf_jobs(status, priority DESC, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_admin_jobs_admin_status ON admin_pdf_jobs(admin_id, status);
                CREATE INDEX IF NOT EXISTS idx_admin_doc_status_admin ON admin_document_status(admin_id, overall_status);
                CREATE INDEX IF NOT EXISTS idx_admin_batches_admin_status ON admin_job_batches(admin_id, status);
            """)
            
            self.logger.info("Admin job tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create job tables: {e}")
            raise
        finally:
            await conn.close()
    
    async def enqueue_admin_pdf_batch(self, admin_id: str, pdf_files: List[Dict]) -> str:
        """
        Enqueue multiple PDF files as a batch for admin processing
        pdf_files: List of {'filename': str, 'file_path': str} dicts
        """
        batch_id = f"batch_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        conn = await self.get_db_connection()
        
        try:
            async with conn.transaction():
                # Create batch record
                await conn.execute("""
                    INSERT INTO admin_job_batches (batch_id, admin_id, total_jobs)
                    VALUES ($1, $2, $3)
                """, batch_id, admin_id, len(pdf_files) * 2)  # 2 jobs per PDF (embedding + summarization)
                
                jobs = []
                for pdf_file in pdf_files:
                    filename = pdf_file['filename']
                    file_path = pdf_file['file_path']
                    
                    # Create document status record
                    await conn.execute("""
                        INSERT INTO admin_document_status 
                        (admin_id, filename, batch_id, processing_start_time)
                        VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                        ON CONFLICT (admin_id, filename, batch_id) DO UPDATE SET
                        embedding_status = 'QUEUED',
                        summarization_status = 'QUEUED', 
                        overall_status = 'QUEUED',
                        updated_at = CURRENT_TIMESTAMP
                    """, admin_id, filename, batch_id)
                    
                    # Create embedding job
                    embedding_job_id = f"emb_{uuid.uuid4().hex[:12]}"
                    embedding_job = AdminJob(
                        id=embedding_job_id,
                        admin_id=admin_id,
                        filename=filename,
                        file_path=file_path,
                        job_type='EMBEDDING',
                        status=JobStatus.QUEUED,
                        priority=JobPriority.ADMIN,
                        created_at=datetime.now(),
                        metadata={'batch_id': batch_id, 'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0}
                    )
                    
                    # Create summarization job
                    summarization_job_id = f"sum_{uuid.uuid4().hex[:12]}"
                    summarization_job = AdminJob(
                        id=summarization_job_id,
                        admin_id=admin_id,
                        filename=filename,
                        file_path=file_path,
                        job_type='SUMMARIZATION',
                        status=JobStatus.QUEUED,
                        priority=JobPriority.ADMIN,
                        created_at=datetime.now(),
                        metadata={'batch_id': batch_id, 'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0}
                    )
                    
                    # Insert jobs into database
                    for job in [embedding_job, summarization_job]:
                        await conn.execute("""
                            INSERT INTO admin_pdf_jobs 
                            (id, admin_id, filename, file_path, job_type, status, priority, 
                             created_at, metadata, batch_id)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        """, job.id, job.admin_id, job.filename, job.file_path, 
                             job.job_type, job.status.value, job.priority.value,
                             job.created_at, json.dumps(job.metadata), batch_id)
                    
                    # Update document status with job IDs
                    await conn.execute("""
                        UPDATE admin_document_status 
                        SET embedding_job_id = $1, summarization_job_id = $2, updated_at = CURRENT_TIMESTAMP
                        WHERE admin_id = $3 AND filename = $4 AND batch_id = $5
                    """, embedding_job_id, summarization_job_id, admin_id, filename, batch_id)
                    
                    jobs.extend([embedding_job, summarization_job])
                
                # Cache jobs and add to queue
                for job in jobs:
                    self.job_status_cache[job.id] = job
                    # Add to admin priority queue (higher priority jobs first)
                    self.admin_queue.put((-job.priority.value, job.created_at.timestamp(), job))
                
                # Create batch object
                batch = JobBatch(
                    batch_id=batch_id,
                    admin_id=admin_id,
                    jobs=jobs,
                    created_at=datetime.now(),
                    total_jobs=len(jobs)
                )
                self.active_batches[batch_id] = batch
                
                self.logger.info(f"Admin {admin_id} queued batch {batch_id} with {len(pdf_files)} PDFs ({len(jobs)} jobs)")
                return batch_id
                
        except Exception as e:
            self.logger.error(f"Failed to enqueue admin PDF batch: {e}")
            raise
        finally:
            await conn.close()
    
    async def enqueue_single_admin_pdf(self, admin_id: str, filename: str, file_path: str) -> str:
        """Quick method to enqueue a single PDF for admin"""
        return await self.enqueue_admin_pdf_batch(admin_id, [{'filename': filename, 'file_path': file_path}])
    
    def get_next_admin_job(self) -> Optional[AdminJob]:
        """Get next admin job from priority queue (non-blocking)"""
        try:
            if not self.admin_queue.empty():
                _, _, job = self.admin_queue.get_nowait()
                # Check if admin is paused
                if job.admin_id not in self.paused_admins:
                    return job
                else:
                    # Put job back if admin is paused
                    self.admin_queue.put((-job.priority.value, job.created_at.timestamp(), job))
            return None
        except queue.Empty:
            return None
    
    def get_next_general_job(self) -> Optional[AdminJob]:
        """Get next general job from priority queue (non-blocking)"""
        try:
            if not self.general_queue.empty():
                _, _, job = self.general_queue.get_nowait()
                return job
            return None
        except queue.Empty:
            return None
    
    async def update_job_status(self, job_id: str, status: JobStatus, error_message: Optional[str] = None):
        """Update job status in database and cache"""
        conn = await self.get_db_connection()
        try:
            async with conn.transaction():
                # Update database
                if status == JobStatus.PROCESSING:
                    await conn.execute("""
                        UPDATE admin_pdf_jobs 
                        SET status = $1, started_at = CURRENT_TIMESTAMP
                        WHERE id = $2
                    """, status.value, job_id)
                elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    actual_duration = None
                    if job_id in self.job_status_cache:
                        job = self.job_status_cache[job_id]
                        if job.started_at:
                            actual_duration = int((datetime.now() - job.started_at).total_seconds())
                    
                    await conn.execute("""
                        UPDATE admin_pdf_jobs 
                        SET status = $1, completed_at = CURRENT_TIMESTAMP, 
                            error_message = $2, actual_duration = $3
                        WHERE id = $4
                    """, status.value, error_message, actual_duration, job_id)
                    
                    # Update batch progress
                    batch_info = await conn.fetchrow("""
                        SELECT batch_id FROM admin_pdf_jobs WHERE id = $1
                    """, job_id)
                    
                    if batch_info and batch_info['batch_id']:
                        if status == JobStatus.COMPLETED:
                            await conn.execute("""
                                UPDATE admin_job_batches 
                                SET completed_jobs = completed_jobs + 1
                                WHERE batch_id = $1
                            """, batch_info['batch_id'])
                        elif status == JobStatus.FAILED:
                            await conn.execute("""
                                UPDATE admin_job_batches 
                                SET failed_jobs = failed_jobs + 1
                                WHERE batch_id = $1
                            """, batch_info['batch_id'])
                
                # Update job cache
                if job_id in self.job_status_cache:
                    self.job_status_cache[job_id].status = status
                    if status == JobStatus.PROCESSING:
                        self.job_status_cache[job_id].started_at = datetime.now()
                    elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                        self.job_status_cache[job_id].completed_at = datetime.now()
                        self.job_status_cache[job_id].error_message = error_message
                
                # Update document status
                job_info = await conn.fetchrow("""
                    SELECT admin_id, filename, job_type, batch_id FROM admin_pdf_jobs WHERE id = $1
                """, job_id)
                
                if job_info:
                    await self.update_document_status(
                        job_info['admin_id'], 
                        job_info['filename'], 
                        job_info['job_type'], 
                        status,
                        job_info['batch_id']
                    )
                
        except Exception as e:
            self.logger.error(f"Failed to update job status for {job_id}: {e}")
        finally:
            await conn.close()
    
    async def update_document_status(self, admin_id: str, filename: str, job_type: str, 
                                   job_status: JobStatus, batch_id: str):
        """Update document processing status"""
        conn = await self.get_db_connection()
        try:
            # Update specific job type status
            if job_type == 'EMBEDDING':
                await conn.execute("""
                    UPDATE admin_document_status 
                    SET embedding_status = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE admin_id = $2 AND filename = $3 AND batch_id = $4
                """, job_status.value, admin_id, filename, batch_id)
            elif job_type == 'SUMMARIZATION':
                await conn.execute("""
                    UPDATE admin_document_status 
                    SET summarization_status = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE admin_id = $2 AND filename = $3 AND batch_id = $4
                """, job_status.value, admin_id, filename, batch_id)
            
            # Check overall document status
            doc_status = await conn.fetchrow("""
                SELECT embedding_status, summarization_status FROM admin_document_status 
                WHERE admin_id = $1 AND filename = $2 AND batch_id = $3
            """, admin_id, filename, batch_id)
            
            if doc_status:
                overall_status = 'PROCESSING'
                if (doc_status['embedding_status'] == 'COMPLETED' and 
                    doc_status['summarization_status'] == 'COMPLETED'):
                    overall_status = 'READY'
                    processing_end_time = datetime.now()
                    await conn.execute("""
                        UPDATE admin_document_status 
                        SET processing_end_time = $1
                        WHERE admin_id = $2 AND filename = $3 AND batch_id = $4
                    """, processing_end_time, admin_id, filename, batch_id)
                elif (doc_status['embedding_status'] == 'FAILED' or 
                      doc_status['summarization_status'] == 'FAILED'):
                    overall_status = 'FAILED'
                
                await conn.execute("""
                    UPDATE admin_document_status 
                    SET overall_status = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE admin_id = $2 AND filename = $3 AND batch_id = $4
                """, overall_status, admin_id, filename, batch_id)
                
        except Exception as e:
            self.logger.error(f"Failed to update document status: {e}")
        finally:
            await conn.close()
    
    def admin_worker(self, worker_id: int):
        """Dedicated worker for admin jobs with higher priority"""
        self.logger.info(f"Admin worker {worker_id} started")
        
        while self.running:
            try:
                job = self.get_next_admin_job()
                
                if job:
                    # Add to running jobs tracking
                    self.admin_running_jobs.add(job.id)
                    
                    self.logger.info(f"Admin worker {worker_id} processing job {job.id}: {job.job_type} for {job.filename}")
                    
                    # Update status to processing
                    asyncio.run(self.update_job_status(job.id, JobStatus.PROCESSING))
                    
                    # Execute job handler
                    if job.job_type in self.job_handlers:
                        try:
                            handler = self.job_handlers[job.job_type]
                            success = asyncio.run(handler(job))
                            
                            if success:
                                asyncio.run(self.update_job_status(job.id, JobStatus.COMPLETED))
                                self.logger.info(f"Admin job {job.id} completed successfully")
                            else:
                                asyncio.run(self.update_job_status(job.id, JobStatus.FAILED, "Handler returned False"))
                                self.logger.warning(f"Admin job {job.id} failed - handler returned False")
                                
                        except Exception as e:
                            error_msg = str(e)
                            self.logger.error(f"Admin job {job.id} failed with exception: {e}")
                            asyncio.run(self.update_job_status(job.id, JobStatus.FAILED, error_msg))
                    else:
                        error_msg = f"No handler for job type: {job.job_type}"
                        self.logger.error(error_msg)
                        asyncio.run(self.update_job_status(job.id, JobStatus.FAILED, error_msg))
                    
                    # Remove from running jobs tracking
                    self.admin_running_jobs.discard(job.id)
                    
                else:
                    # No admin jobs available, sleep briefly
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Admin worker {worker_id} error: {e}")
                time.sleep(2)
        
        self.logger.info(f"Admin worker {worker_id} stopped")
    
    def general_worker(self, worker_id: int):
        """General worker for non-admin jobs"""
        self.logger.info(f"General worker {worker_id} started")
        
        while self.running:
            try:
                job = self.get_next_general_job()
                
                if job:
                    # Add to running jobs tracking
                    self.running_jobs.add(job.id)
                    
                    self.logger.info(f"General worker {worker_id} processing job {job.id}: {job.job_type}")
                    
                    # Same processing logic as admin worker
                    asyncio.run(self.update_job_status(job.id, JobStatus.PROCESSING))
                    
                    if job.job_type in self.job_handlers:
                        try:
                            handler = self.job_handlers[job.job_type]
                            success = asyncio.run(handler(job))
                            
                            status = JobStatus.COMPLETED if success else JobStatus.FAILED
                            error_msg = None if success else "Handler returned False"
                            asyncio.run(self.update_job_status(job.id, status, error_msg))
                            
                        except Exception as e:
                            asyncio.run(self.update_job_status(job.id, JobStatus.FAILED, str(e)))
                    else:
                        asyncio.run(self.update_job_status(job.id, JobStatus.FAILED, f"No handler for job type: {job.job_type}"))
                    
                    self.running_jobs.discard(job.id)
                    
                else:
                    time.sleep(2)  # Sleep longer for general jobs
                    
            except Exception as e:
                self.logger.error(f"General worker {worker_id} error: {e}")
                time.sleep(3)
        
        self.logger.info(f"General worker {worker_id} stopped")
    
    def monitor_jobs(self):
        """Monitor job progress and handle stalled jobs"""
        self.logger.info("Job monitor started")
        
        while self.running:
            try:
                # Check for stalled jobs (processing for too long)
                current_time = datetime.now()
                stalled_threshold = timedelta(minutes=30)  # Adjust as needed
                
                for job_id in list(self.admin_running_jobs) + list(self.running_jobs):
                    if job_id in self.job_status_cache:
                        job = self.job_status_cache[job_id]
                        if (job.started_at and 
                            current_time - job.started_at > stalled_threshold):
                            self.logger.warning(f"Job {job_id} appears stalled, marking as failed")
                            asyncio.run(self.update_job_status(job_id, JobStatus.FAILED, "Job stalled - exceeded time limit"))
                
                # Update batch completion status
                for batch_id, batch in self.active_batches.items():
                    conn = asyncio.run(self.get_db_connection())
                    try:
                        batch_status = asyncio.run(conn.fetchrow("""
                            SELECT completed_jobs, failed_jobs, total_jobs FROM admin_job_batches 
                            WHERE batch_id = $1
                        """, batch_id))
                        
                        if batch_status:
                            total_processed = batch_status['completed_jobs'] + batch_status['failed_jobs']
                            if total_processed >= batch_status['total_jobs']:
                                asyncio.run(conn.execute("""
                                    UPDATE admin_job_batches 
                                    SET status = 'COMPLETED', completed_at = CURRENT_TIMESTAMP
                                    WHERE batch_id = $1
                                """, batch_id))
                                
                                self.logger.info(f"Batch {batch_id} completed: {batch_status['completed_jobs']} successful, {batch_status['failed_jobs']} failed")
                    finally:
                        await conn.close()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Job monitor error: {e}")
                time.sleep(60)
        
        self.logger.info("Job monitor stopped")
    
    def start(self):
        """Start the admin job queue manager"""
        self.running = True
        
        # Initialize database tables
        asyncio.run(self.create_job_tables())
        
        # Start admin workers (dedicated high-priority workers)
        for i in range(self.admin_workers):
            worker = threading.Thread(target=self.admin_worker, args=(f"admin-{i}",))
            worker.daemon = True
            worker.start()
        
        # Start general workers
        for i in range(self.max_workers):
            worker = threading.Thread(target=self.general_worker, args=(f"general-{i}",))
            worker.daemon = True
            worker.start()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_jobs)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info(f"Admin Job Queue Manager started: {self.admin_workers} admin workers, {self.max_workers} general workers")
    
    def stop(self):
        """Stop the admin job queue manager"""
        self.running = False
        
        # Wait for threads to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Shutdown executors
        self.admin_executor.shutdown(wait=True)
        self.general_executor.shutdown(wait=True)
        
        self.logger.info("Admin Job Queue Manager stopped")
    
    async def get_admin_status(self, admin_id: str) -> Dict:
        """Get comprehensive status for admin"""
        conn = await self.get_db_connection()
        try:
            # Get batch information
            batches = await conn.fetch("""
                SELECT * FROM admin_job_batches 
                WHERE admin_id = $1 
                ORDER BY created_at DESC 
                LIMIT 10
            """, admin_id)
            
            # Get document status
            documents = await conn.fetch("""
                SELECT * FROM admin_document_status 
                WHERE admin_id = $1 
                ORDER BY created_at DESC
            """, admin_id)
            
            # Get active jobs
            active_jobs = await conn.fetch("""
                SELECT * FROM admin_pdf_jobs 
                WHERE admin_id = $1 AND status IN ('QUEUED', 'PROCESSING')
                ORDER BY priority DESC, created_at ASC
            """, admin_id)
            
            # Calculate statistics
            total_documents = len(documents)
            ready_documents = len([d for d in documents if d['overall_status'] == 'READY'])
            processing_documents = len([d for d in documents if d['overall_status'] == 'PROCESSING'])
            failed_documents = len([d for d in documents if d['overall_status'] == 'FAILED'])
            
            return {
                'admin_id': admin_id,
                'batches': [dict(b) for b in batches],
                'documents': [dict(d) for d in documents],
                'active_jobs': [dict(j) for j in active_jobs],
                'statistics': {
                    'total_documents': total_documents,
                    'ready_documents': ready_documents,
                    'processing_documents': processing_documents,
                    'failed_documents': failed_documents,
                    'queue_length': len(active_jobs),
                    'success_rate': round((ready_documents / total_documents * 100) if total_documents > 0 else 0, 2)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get admin status: {e}")
            return {'error': str(e)}
        finally:
            await conn.close()
    
    def pause_admin_jobs(self, admin_id: str):
        """Pause job processing for specific admin (emergency stop)"""
        self.paused_admins.add(admin_id)
        self.logger.info(f"Paused jobs for admin {admin_id}")
    
    def resume_admin_jobs(self, admin_id: str):
        """Resume job processing for specific admin"""
        self.paused_admins.discard(admin_id)
        self.logger.info(f"Resumed jobs for admin {admin_id}")
    
    async def get_batch_progress(self, batch_id: str) -> Dict:
        """Get detailed progress for a specific batch"""
        conn = await self.get_db_connection()
        try:
            batch_info = await conn.fetchrow("""
                SELECT * FROM admin_job_batches WHERE batch_id = $1
            """, batch_id)
            
            if not batch_info:
                return {'error': 'Batch not found'}
            
            jobs = await conn.fetch("""
                SELECT * FROM admin_pdf_jobs WHERE batch_id = $1 ORDER BY created_at
            """, batch_id)
            
            documents = await conn.fetch("""
                SELECT * FROM admin_document_status WHERE batch_id = $1 ORDER BY created_at
            """, batch_id)
            
            return {
                'batch': dict(batch_info),
                'jobs': [dict(j) for j in jobs],
                'documents': [dict(d) for d in documents],
                'progress': {
                    'total': batch_info['total_jobs'],
                    'completed': batch_info['completed_jobs'],
                    'failed': batch_info['failed_jobs'],
                    'remaining': batch_info['total_jobs'] - batch_info['completed_jobs'] - batch_info['failed_jobs'],
                    'percentage': round(((batch_info['completed_jobs'] + batch_info['failed_jobs']) / batch_info['total_jobs'] * 100), 2)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get batch progress: {e}")
            return {'error': str(e)}
        finally:
            await conn.close()