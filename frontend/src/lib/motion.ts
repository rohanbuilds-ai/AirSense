/**
 * Shared Motion Design Tokens & Animation Variants for AirSense.
 * Built with motion/react, adhering to scientific precision and reduced-motion guidelines.
 */

import { Variants } from 'motion/react';

// Transition easings
export const EASINGS = {
  easeOutCubic: [0.33, 1, 0.68, 1],
  easeInOutCubic: [0.65, 0, 0.35, 1],
  gentleSpring: { type: 'spring', stiffness: 380, damping: 30 },
  snappySpring: { type: 'spring', stiffness: 450, damping: 25 },
} as const;

// Page & Container Entrance Variants
export const pageVariants: Variants = {
  initial: { opacity: 0, y: 8 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.28, ease: EASINGS.easeOutCubic },
  },
  exit: {
    opacity: 0,
    y: -4,
    transition: { duration: 0.16, ease: 'easeIn' },
  },
};

export const staggerContainer: Variants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.04,
      delayChildren: 0.02,
    },
  },
};

export const staggerItem: Variants = {
  initial: { opacity: 0, y: 10 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.24, ease: EASINGS.easeOutCubic },
  },
};

export const cardHoverVariants: Variants = {
  rest: {
    y: 0,
    boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    transition: { duration: 0.18, ease: 'easeOut' },
  },
  hover: {
    y: -2,
    boxShadow: '0 8px 16px -4px rgba(15, 23, 42, 0.08), 0 2px 4px -2px rgba(15, 23, 42, 0.04)',
    transition: { duration: 0.18, ease: 'easeOut' },
  },
};

export const tabContentVariants: Variants = {
  initial: { opacity: 0, y: 6 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.2, ease: EASINGS.easeOutCubic },
  },
  exit: {
    opacity: 0,
    y: -4,
    transition: { duration: 0.14, ease: 'easeIn' },
  },
};

export const accordionVariants: Variants = {
  collapsed: {
    opacity: 0,
    height: 0,
    transition: { duration: 0.22, ease: 'easeInOut' },
  },
  expanded: {
    opacity: 1,
    height: 'auto',
    transition: { duration: 0.28, ease: EASINGS.easeOutCubic },
  },
};

export const modalBackdropVariants: Variants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.18 } },
  exit: { opacity: 0, transition: { duration: 0.14 } },
};

export const modalContentVariants: Variants = {
  initial: { opacity: 0, scale: 0.97, y: 8 },
  animate: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: { duration: 0.24, ease: EASINGS.easeOutCubic },
  },
  exit: {
    opacity: 0,
    scale: 0.98,
    y: 4,
    transition: { duration: 0.16, ease: 'easeIn' },
  },
};
