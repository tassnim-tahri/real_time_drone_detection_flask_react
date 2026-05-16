import express from "express";
import User from "../models/User.js";

const router = express.Router();

router.post("/register", async (req, res) => {
  try {
    const { username, password } = req.body;

    // 1️⃣ Check if user exists
    const existingUser = await User.findOne({ username });

    if (existingUser) {
      return res.status(400).json({ message: "User already exists" });
    }

    // 2️⃣ Create user
    const newUser = new User({
      username,
      password,
      verified: false,
      state: "pending"
    });

    await newUser.save();

    // 3️⃣ Send response FIRST
    res.json({ message: "User registered successfully" });

    // 4️⃣ (Optional) Log for debugging
    console.log("New user created:", username);

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
