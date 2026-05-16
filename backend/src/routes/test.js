import express from "express";
import User from "../models/User.js";

const router = express.Router();

// GET all users
router.get("/users", async (req, res) => {
  try {
    const users = await User.find();
    res.json(users);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// CREATE test user
router.get("/create", async (req, res) => {
  try {
    const user = new User({
      username: "test-test",
      password: "1234"
    });

    await user.save();

    res.json({ message: "User created", user });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
