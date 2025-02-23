module ALU(
	input clear,
	input clock,
	input wire [4:0] opcode,
	input wire [31:0] A,
	input wire [31:0] B,
	output reg [63:0] Z,
	output reg [31:0] Zhighout,
	output reg [31:0] Zlowout
);

	parameter Logical_AND = 5'b00101, Logical_OR = 5'b00110, Addition = 5'b00011, Subtraction = 5'b00100, Multiply = 5'b10000, Division = 5'b01111,
	Shift_R = 5'b01001, Shift_Right_A = 5'b01010, Shift_L = 5'b01011, Rotate_R = 5'b00111, Rotate_L = 5'b01000, Negate = 5'b10001, Not = 5'b10010;

	wire [31:0] and_result, or_result, add_result, sub_result, shr_result, shra_result, shl_result, ror_result, rol_result, neg_result, not_result;
	wire [63:0] mul_result, div_result;


	always @(*) 
		begin
			case (opcode)

				Logical_AND: begin // 3.1
					Z[31:0] <= and_result[31:0];
					Z[63:32] <= 32'd0;
					
				end
				
				Logical_OR: begin // 3.2
					Z[31:0] <= or_result[31:0];
					
				end
				
				Addition: begin // 3.3
					Z[31:0] <= add_result[31:0];
					Z[63:32] <= 32'd0;
					
				end
				
				Subtraction: begin // 3.4
					Z[31:0] <= sub_result[31:0];
					
				end
				
				Multiply: begin // 3.5
					Zlowout[31:0] <= mul_result[31:0];
					Zhighout[31:0] <= mul_result[63:32];
				end
				
				Division: begin // 3.6
					Zlowout[31:0] <= div_result[31:0];
					Zhighout[31:0] <= div_result[63:32];
				end
				
				Shift_R: begin // 3.7
					Z[31:0] <= shr_result[31:0];
					
				end
				
				Shift_Right_A: begin // 3.8
					Z[31:0] <= shra_result[31:0];
					
				end
				
				Shift_L: begin // 3.9
					Z[31:0] <= shl_result[31:0];
					
				end

				Rotate_R: begin // 3.10
					Z[31:0] <= ror_result[31:0];
					
				end
	
				Rotate_L: begin // 3.11
					Z[31:0] <= rol_result[31:0];
					
				end
				
				Negate: begin // 3.12
					Z[31:0] <= neg_result[31:0];
					
				end

				Not: begin // 3.13
					Z[31:0] <= not_result[31:0];
					
				end
				
			endcase
	end

	logicalAND logicalAnd(A, B, and_result);
	//logical_OR 	logical_or(A, B, or_result);
	RCA 		add(A, B, add_result);
	sub 	sub(A, B, sub_result);
	rightShift rightShift(A, B, shr_result);
	rightShiftA rightShiftA(A, B, shra_result);
	leftShift leftShift(A, B, shl_result);
	rightRotate rightRotate(A, B, ror_result);
	leftRotate leftRotate(A, B, rol_result);
	neg	neg(A, neg_result);
	
//	booth 		mul(A, B, mul_result[31:0], mul_result[63:32]);
//	division		div(A, B, div_result[31:0], div_result[63:32]);
//	logical_NOT logical_not(B, not_result);


endmodule