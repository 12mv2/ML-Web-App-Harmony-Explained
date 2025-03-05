import { Request, Response, NextFunction } from 'express';
import { uploadAudio, generateSignedUrl } from '../model/storage';
import { supabase } from '../model/db';
import pool from '../model/db';
import * as fs from 'node:fs/promises';
import { CustomError } from '../types';

const DEFAULT_TIMEOUT_MS = 3 * 60 * 60 * 1000; // 20 minutes
const ML_SERVICE_TIMEOUT_MS = Number(process.env.ML_SERVICE_TIMEOUT_MS) || DEFAULT_TIMEOUT_MS;

/**
 * Helper function to add a timeout to fetch, with detailed logging.
 */
const fetchWithTimeout = async (url: string, options: RequestInit, timeout: number = ML_SERVICE_TIMEOUT_MS) => {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  const start = Date.now();
  console.log(`[fetchWithTimeout] Starting request to ${url} with timeout ${timeout / 1000} seconds`);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });

    clearTimeout(id);
    const duration = (Date.now() - start) / 1000;
    console.log(`[fetchWithTimeout] Request to ${url} completed in ${duration} seconds`);

    return response;
  } catch (err) {
    clearTimeout(id);
    const duration = (Date.now() - start) / 1000;

    if (err instanceof Error && err.name === 'AbortError') {
      console.error(`[fetchWithTimeout] Request to ${url} timed out after ${duration} seconds`);
      throw new Error(`Request to ML service timed out after ${duration} seconds`);
    }

    console.error(`[fetchWithTimeout] Request to ${url} failed after ${duration} seconds:`, err);

    if (err instanceof TypeError && err.message.includes('fetch failed')) {
      console.error(`[fetchWithTimeout] Network-level error detected (DNS failure, connection refused, or ML service down).`);
    }

    throw err;
  }
};

const audioController = {
  transform: async (req: Request, res: Response, next: NextFunction) => {
    console.log('Transform endpoint hit');
    console.log('Request file metadata:', req.file);

    if (!req.file) {
      const err: CustomError = {
        log: 'No file uploaded for transformation',
        status: 400,
        message: { err: 'No file uploaded' },
      };
      return next(err);
    }

    try {
      console.log(`Reading uploaded file: ${req.file.path}`);
      const fileContent = await fs.readFile(req.file.path);
      console.log(`File read successfully, size: ${fileContent.length} bytes`);

      console.log('Preparing form data to send to ML service...');
      const formData = new FormData();
      formData.append(
        'audio_file',
        new File([fileContent], req.file.originalname, { type: req.file.mimetype })
      );

      console.log(`Sending request to ML service at http://localhost:8000/generate with timeout ${ML_SERVICE_TIMEOUT_MS / 1000} seconds`);

      const mlResponse = await fetchWithTimeout('http://localhost:8000/generate', {
        method: 'POST',
        body: formData,
      }, ML_SERVICE_TIMEOUT_MS);

      if (!mlResponse.ok) {
        console.error(`ML service responded with non-200 status: ${mlResponse.status}`);
        throw new Error(`ML service returned error: ${mlResponse.status} ${mlResponse.statusText}`);
      }

      const transformedBuffer = Buffer.from(await mlResponse.arrayBuffer());
      console.log(`Received transformed audio file from ML service, size: ${transformedBuffer.length} bytes`);

      res.set('Content-Type', 'audio/wav');
      res.send(transformedBuffer);
    } catch (error) {
      console.error('Error during audio transformation:', error);

      const err: CustomError = {
        log: `Audio transformation failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: 500,
        message: { err: 'Audio transformation failed' },
      };
      next(err);
    } finally {
      if (req.file?.path) {
        console.log(`Cleaning up temporary uploaded file: ${req.file.path}`);
        await fs.unlink(req.file.path).catch((cleanupErr) => {
          console.error('Failed to delete temporary file:', cleanupErr);
        });
      }
    }
  },

  upload: async (req: Request, res: Response, next: NextFunction) => {
    console.log('Upload request received.');
    console.log('Upload request body:', req.body);
    console.log('Files received:', req.files);
    console.log('Authenticated user:', req.user);

    if (!req.files || !req.user) {
      const err: CustomError = {
        log: 'Validation failed - Missing files or user authentication',
        status: 400,
        message: { err: 'Files or authentication data missing' },
      };
      return next(err);
    }

    const files = req.files as {
      originalFile: Express.Multer.File[];
      transformedFile: Express.Multer.File[];
    };

    try {
      const originalFilePath = `${req.user.id}/${Date.now()}_${files.originalFile[0].originalname}`;
      const transformedFilePath = `${req.user.id}/${Date.now()}_transformed_${files.transformedFile[0].originalname}`;

      console.log('Uploading original file to Supabase...');
      const { error: originalError } = await supabase.storage
        .from('original-audio')
        .upload(originalFilePath, await fs.readFile(files.originalFile[0].path), {
          contentType: files.originalFile[0].mimetype,
        });

      if (originalError) throw originalError;

      const originalUrl = await generateSignedUrl('original-audio', originalFilePath);
      console.log('Original file uploaded successfully.');

      console.log('Uploading transformed file to Supabase...');
      const { error: transformedError } = await supabase.storage
        .from('transformed-audio')
        .upload(transformedFilePath, await fs.readFile(files.transformedFile[0].path), {
          contentType: 'audio/wav',
        });

      if (transformedError) throw transformedError;

      const transformedUrl = await generateSignedUrl('transformed-audio', transformedFilePath);
      console.log('Transformed file uploaded successfully.');

      res.json({
        success: true,
        data: { originalUrl, transformedUrl },
      });
    } catch (error) {
      console.error('Error in upload handler:', error);
      next(error);
    } finally {
      console.log('Starting cleanup of local uploaded files...');
      if (files.originalFile?.[0]?.path) {
        await fs.unlink(files.originalFile[0].path).catch((err) =>
          console.error('Error cleaning original file:', err)
        );
      }
      if (files.transformedFile?.[0]?.path) {
        await fs.unlink(files.transformedFile[0].path).catch((err) =>
          console.error('Error cleaning transformed file:', err)
        );
      }
    }
  },
};

export default audioController;
