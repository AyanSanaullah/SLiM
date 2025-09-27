import type { Metadata } from "next";
import { Geist, Geist_Mono, Turret_Road } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const turretRoad = Turret_Road({
  variable: "--font-turret-road",
  subsets: ["latin"],
  weight: ["200", "300", "400", "500", "700", "800"],
});

export const metadata: Metadata = {
  title: "Sum - AI Assistant",
  description: "Where saving energy goes hand in hand with creating better quality results",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${turretRoad.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
