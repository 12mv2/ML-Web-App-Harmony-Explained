import { Request, Response } from "express";
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken'
import { supabase } from '../model/db';

// Development mode flag - we can change this to false when ready to use Supabase
const USE_DEV_AUTH = true;

// Test user for development
const TEST_USER = {
  id: 1,
  username: '123',
  password: '123',
  email: 'test@test.com'
};

const JWT_SECRET_KEY = process.env.JWT_SECRET_KEY || 'your_secret_key'; 

const authController = {
    login: async (req: Request, res: Response) => {
      console.log('Login request body: ', req.body);
      const {username, password} = req.body;

      try {
        console.log('Login attempt for:', username);
        if (!username || !password) {
          return res.status(400).json({ 
            success: false,
            message: 'All fields are required.' 
          });
        }

        // Development auth path
        if (USE_DEV_AUTH) {
          if (username === TEST_USER.username && password === TEST_USER.password) {
            const token = jwt.sign(
              { id: TEST_USER.id, username: TEST_USER.username },
              JWT_SECRET_KEY,
              { expiresIn: '1h' }
            );
            
            return res.status(200).json({ 
              success: true,
              message: 'Login successful', 
              token,
              user: { id: TEST_USER.id, username: TEST_USER.username }
            });
          }
          return res.status(400).json({
            success: false,
            message: 'Invalid credentials'
          });
        }
        
        // Production auth path (unchanged)
        const {data: user, error} = await supabase
          .from('users')
          .select('id, username, password')
          .eq('username', username)
          .single();

        if (error || !user) {
          return res.status(400).json({ 
            success: false,
            message: 'User not found'
          });
        }

        const passwordMatched = await bcrypt.compare(password, user.password);
        if (!passwordMatched) {
          return res.status(400).json({
            success: false,
            message: 'Invalid credentials'
          });
        }

        const token = jwt.sign(
          {id: user.id, username: user.username },
          JWT_SECRET_KEY,
          { expiresIn: '1h' }
        );

        return res.status(200).json({ 
          success: true,
          message: 'Login successful', 
          token,
          user: { id: user.id, username: user.username }
        });
  
      } catch (error) {
        console.error('Error logging in user:', error);
        res.status(500).json({ 
          success: false,
          message: 'Internal server error'
        });
      }
    },

    register: async (req: Request, res: Response) => {
      const { username, email, password } = req.body;
      console.log('checking req body', username, email, password);

      try {
        // Development register path
        if (USE_DEV_AUTH) {
          return res.status(201).json({ 
            message: 'User created successfully', 
            user: TEST_USER 
          });
        }

        // Production register path (unchanged)
        const { data: existingUser } = await supabase
          .from('users')
          .select('id')
          .eq('email', email)
          .single();

        if (existingUser) {
          return res.status(400).json({ message: 'Email is already in use.'});
        }
        
        const salt = 10;
        const hashedPassword = await bcrypt.hash(password, salt);

        const { data:user, error } = await supabase
          .from('users')
          .insert([{ username, email, password: hashedPassword }]);

        if (error) {
          return res.status(500).json({ message: 'Failed to create user', error });
        }

        return res.status(201).json({ message: 'User created successfully', user: user });

      } catch (error) {
        console.error('Error creating user:', error);
        res.status(500).json({ message: 'Internal server error', error});
      }
    }
};

export default authController;

// import { Request, Response } from "express";
// import bcrypt from 'bcrypt';
// import jwt from 'jsonwebtoken'
// // import pool from '../model/db'
// import { supabase } from '../model/db';


// const JWT_SECRET_KEY = process.env.JWT_SECRET_KEY || 'your_secret_key'; 

// const authController = {

//     login: async (req: Request, res: Response) => {
//       console.log('Login request body: ', req.body);
//       const {username, password} = req.body
//       try {
//         console.log('Login attempt for:', username); // Debug log
//          if (!username || !password) {
//             return res.status(400).json({ 
//               success: false,
//               message: 'All fields are required.' });
//           }
        
//       // Find the user by username 
//       const {data: user, error} = await supabase
//       .from('users')
//       .select('id, username, password')
//       .eq('username', username)
//       .single();

//       if (error || !user) {
//         return res.status(400).json({ 
//           success: false,
//           message: 'User not found'
//         });
//       }

//       // Compare the password with the hashed password
//       const passwordMatched = await bcrypt.compare(password, user.password)

//       if (!passwordMatched) {
//         return res.status(400).json({
//           success: false,
//           message: 'Insvalid credential' 
//         })
//       }

//       // Create a JWT token after successful login
//       const token = jwt.sign(
//         {id: user.id, username: user.username },
//         JWT_SECRET_KEY, // Secret key used to sign the token
//         { expiresIn: '1h' } // Token expiration time (1hour)
//       );

//       console.log('Token generated:', token); // Debug log
//       return res.status(200).json({ 
//         success: true,
//         message: 'Login successful', 
//         token,
//         user: { id: user.id, username: user.username }
//       });  
  
//       } catch (error) {
//       console.error('Error loggin in user:', error);
//       res.status(500).json({ 
//         success: false,
//         message: 'Internal server error'
//         });
//       }
//     },

//     register: async (req: Request, res: Response) => {
//       const { username, email, password } = req.body;
//       console.log('checking req body', username, email, password)
//     try {
//       const { data: existingUser } = await supabase
//       .from('users')
//       .select('id')
//       .eq('email', email)
//       .single();

//     if (existingUser) {
//       return res.status(400).json({ message: 'Email is already in use.'});
//     }
    
//     //Hash the password using bcrypt
//     const salt = 10;
//     const hashedPassword = await bcrypt.hash(password, salt)

//     //Insert user into the database
//     const { data:user, error } = await supabase
//     .from('users')
//     .insert([
//       {
//         username,
//         email,
//         password: hashedPassword,
//       },
//     ]);

//     if (error) {
//       return res.status(500).json({ message: 'Failed to create user', error });
//     }

//     return res.status(201).json({ message: 'User created successfully', user: user })

//     }catch (error) {
//       console.error('Error creating user:', error);
//       res.status(500).json({ message: 'Internal server error', error})
//     }
//   }
// }
  
//   export default authController;