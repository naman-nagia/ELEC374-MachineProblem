module not_operation (
    input wire [31:0] A, 
    output wire [31:0] result
);
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : bitwise_not
            assign result[i] = ~A[i]; // Bitwise NOT
        end
    endgenerate
endmodule
