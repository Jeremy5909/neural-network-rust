use std::{fmt, ops::{Add, Sub}};
use rand::prelude::*;

#[derive(Debug, PartialEq, Clone)]
pub struct Matrix {
	pub rows: usize,
	pub cols: usize,
	pub data: Vec<f64>,
}
impl Matrix {
	pub fn random(rows: usize, cols: usize) -> Self {
		let mut buffer: Vec<_> = Vec::<f64>::with_capacity(rows * cols);
		
		for _ in 0..rows*cols {
			let num: f64 = rand::thread_rng().gen_range(0.0..1.0);

			buffer.push(num);
		}
		Matrix { rows, cols, data:buffer }
	}

    pub fn dot_prod(&self, rhs: &Matrix) -> Matrix {
        assert!(self.cols == rhs.rows, "Cannot take the dot product of matrices with wrong dimensions");
       
        let mut result_data = vec![0.0; self.rows * rhs.cols];

        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = 0.0f64;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * rhs.data[k * rhs.cols + j];
                }
                result_data[i * rhs.cols + j] = sum;
            }
        }

        Matrix {
            rows: self.rows,
            cols: rhs.cols,
            data: result_data,
        }
    }
    pub fn map(&mut self, func: fn(&f64) -> f64) -> Matrix {
        let mut result = Matrix {
            rows: self.rows,
            cols: self.cols,
            data: Vec::with_capacity(self.data.len()),
        };

        result.data.extend(self.data.iter().map(|&val| func(&val)));

        result
    }
    pub fn elementwise_multiply(&self, rhs: &Matrix) -> Matrix {
       assert_eq!(self.rows, rhs.rows);
       assert_eq!(self.cols, rhs.cols);
       let mut result_data = vec![0.0; self.cols * self.rows];
       for i in 0..self.data.len() {
           result_data[i] = self.data[i] * rhs.data[i];
       }

       Matrix {
        rows: self.rows,
        cols: self.cols,
        data: result_data,
       }
    }
    pub fn transpose(&self) -> Matrix {
        let mut buffer = vec![0.0; self.cols * self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                buffer[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: buffer,
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.rows {
            for col in 0..self.cols {
                write!(f, "{}", self.data[row * self.cols + col])?;
                if col < self.cols - 1 {
                    write!(f, "\t")?; // Separate columns with a tab
                }
            }
            writeln!(f)?; // Move to the next line after each row
        }
        Ok(())
    }
}

impl From<Vec<f64>> for Matrix {
    fn from(vec: Vec<f64>) -> Self {
        let rows = vec.len();
        let cols = 1;
        Matrix {
            rows,
            cols,
            data: vec
        }
    }
}
impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols, "Cannot add matrices with different dimensions");

        let mut buffer = Vec::<f64>::with_capacity(self.rows * self.cols);

        for i in 0..self.data.len() {
            let result = self.data[i] + rhs.data[i];
            buffer.push(result);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: buffer
        }
    }
}

impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Self) -> Self::Output {
        assert!(self.rows == rhs.rows && self.cols == rhs.cols, "Cannot subtract matrices with different dimensions");
        
        let mut buffer = Vec::<f64>::with_capacity(self.rows * self.cols);

        for i in 0..self.data.len() {
            let result = self.data[i] - rhs.data[i];
            buffer.push(result);
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: buffer
        }
    }
}
