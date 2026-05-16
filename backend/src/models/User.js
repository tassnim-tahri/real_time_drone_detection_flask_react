import mongoose from "mongoose";

const userSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true
  },
  password: {
    type: String,
    required: true
  },
  email: {
    type: String,
    required: true,
    unique: true
  },
  lastLogin: {
    type: Date
  },
  approvedAt: {
    type: Date
  },
  approvedBy: {
    type: String
  },

  age: {
    type: Number,
    min: 0
  },

  verified: {
    type: Boolean,
    default: false
  },
  state: {
    type: String,
    enum: ["pending", "admin", "operator"],
    default: "pending"
  }
}, { timestamps: true });

export default mongoose.model("User", userSchema);

