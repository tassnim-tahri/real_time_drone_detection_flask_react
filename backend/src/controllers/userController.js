import User from "../models/User.js";
import bcrypt from "bcryptjs";
// GET all users
export const getAllUsers = async (req, res) => {
  try {
    const users = await User.find();
    res.json(users);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
};

// UPDATE state (pending → admin/operator)
export const updateUserState = async (req, res) => {
  try {
    const { state } = req.body;

    // ✅ FIX: allow all 3 states
    if (!["admin", "operator"].includes(state)) {
      return res.status(400).json({ message: "Invalid state" });
    }

    const user = await User.findByIdAndUpdate(
      req.params.id,
      {
        state,
        verified: state !== "pending" // optional logic
      },
      { new: true }
    );

    res.json(user);
  } catch (err) {
    console.log(err); // 👈 VERY IMPORTANT for debugging
    res.status(500).json({ message: err.message });
  }
};


export const userDelete = async (req, res) => {
  try {
    const deletedUser = await User.findByIdAndDelete(req.params.id);

    if (!deletedUser) {
      return res.status(404).json({ message: "User not found" });
    }

    res.json({ message: "User deleted successfully" });
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
};

export const getUserByEmail = async (req, res) => {
  try {
    const user = await User.findOne({ email: { $regex: `^${req.params.email}$`, $options: "i" } });
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }
    res.json(user);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
};

export const login = async (req, res) => {
  try {
    const { username, password } = req.body;

    const user = await User.findOne({
      $or: [
        { username:  { $regex: `^${username}$`, $options: "i" } },
        { email:  { $regex: `^${username}$`, $options: "i" } }
      ]
    });

    if (!user) {
      return res.status(400).json({ message: "User not found" });
    }


    // 🔐 Compare hashed password
    const isMatch = await bcrypt.compare(password, user.password);

    if (!isMatch) {
      return res.status(400).json({ message: "Invalid credentials" });
    }

    if (user.state === "pending") {
      return res.status(403).json({ message: "Waiting for admin approval" });
    }

    res.json(user);

  } catch (err) {
    res.status(500).json({ message: err.message });
  }
};


export const register = async (req, res) => {
  try {
    const { username, email, password, age } = req.body;

    // 🔍 Check if user already exists (username OR email)
    const existingUser = await User.findOne({
      $or: [{ username }, { email }]
    });

    if (existingUser) {
      return res.status(400).json({ message: "User already exists" });
    }

    // 🔐 Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    // ✅ Create user
    const newUser = new User({
      username,
      email,
      password: hashedPassword,
      age,
      state: "pending",
      verified: false
    });

    await newUser.save();

    res.status(201).json({ message: "User registered, waiting for approval" });

  } catch (err) {
    res.status(500).json({ message: err.message });
  }
};

export const updateUserPassword = async (req, res) => {
  try {
    const { password } = req.body;
    if (!password) {
      return res.status(400).json({ message: "Password required" });
    }
    //const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, 10);
    const user = await User.findByIdAndUpdate(
      req.params.id,
      { password: hashedPassword },
      { new: true }
    );  
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }
    res.json({ message: "Password updated successfully" });
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
};

export const add_user = async (req, res) => {
  try {
    const { username, email, password, age, state } = req.body;

    // 🔍 Check if user already exists (username OR email)
    const existingUser = await User.findOne({
      $or: [ { username: { $regex: `^${username}$`, $options: "i" } }, { email: { $regex: `^${email}$`, $options: "i" } } ]
    });

    if (existingUser) {
      return res.status(400).json({ message: "User already exists" });
    }

    // 🔐 Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    // ✅ Create user
    const newUser = new User({
      username,
      email,
      password: hashedPassword,
      age,
      state: state || "pending",
      verified: state === "pending" ? false : true
    });

    await newUser.save();

    res.status(201).json({ message: "User added successfully" });

  } catch (err) {
    res.status(500).json({ message: err.message });
  }
};