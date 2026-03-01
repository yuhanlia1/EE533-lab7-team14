`timescale 1ns/1ps

module tb;
  reg clk;
  reg reset;

  reg                        reg_req_in;
  reg                        reg_ack_in;
  reg                        reg_rd_wr_L_in;
  reg  [22:0]                reg_addr_in;
  reg  [31:0]                reg_data_in;
  reg  [1:0]                 reg_src_in;

  wire                       reg_req_out;
  wire                       reg_ack_out;
  wire                       reg_rd_wr_L_out;
  wire [22:0]                reg_addr_out;
  wire [31:0]                reg_data_out;
  wire [1:0]                 reg_src_out;

  localparam [7:0] GPU_PREFIX = 8'h7F;

  localparam OFF_CONTROL     = 8'h00;
  localparam OFF_A_BASE      = 8'h01;
  localparam OFF_B_BASE      = 8'h02;
  localparam OFF_C_BASE      = 8'h03;
  localparam OFF_N_WORDS     = 8'h04;
  localparam OFF_IMEM_ADDR   = 8'h10;
  localparam OFF_IMEM_WDATA  = 8'h11;
  localparam OFF_IMEM_WE     = 8'h12;
  localparam OFF_DMEM_ADDR   = 8'h20;
  localparam OFF_DMEM_WLO    = 8'h21;
  localparam OFF_DMEM_WHI    = 8'h22;
  localparam OFF_DMEM_WE     = 8'h23;
  localparam OFF_DMEM_RLO    = 8'h24;
  localparam OFF_DMEM_RHI    = 8'h25;

  function [22:0] A;
    input [7:0] off;
    begin
      A = {GPU_PREFIX, 7'd0, off};
    end
  endfunction

  function [31:0] I;
    input [3:0] op;
    input [2:0] rd;
    input [2:0] rs1;
    input [2:0] rs2;
    input [1:0] bsel;
    input [15:0] imm16;
    begin
      I = {op, rd, rs1, rs2, bsel, 1'b0, imm16};
    end
  endfunction

  function [31:0] S;
    input [3:0] op;
    input [2:0] rs2;
    input [1:0] bsel;
    input [15:0] imm16;
    begin
      S = {op, 3'd0, 3'd0, rs2, bsel, 1'b0, imm16};
    end
  endfunction

  function [31:0] T;
    input [3:0] op;
    input [9:0] tid;
    begin
      T = {op, 18'd0, tid};
    end
  endfunction

  function [31:0] B;
    input [3:0] op;
    input [7:0] target;
    begin
      B = {op, 20'd0, target};
    end
  endfunction

  gpu_datapath dut(
    .clk(clk),
    .reset(reset),
    .reg_req_in(reg_req_in),
    .reg_ack_in(reg_ack_in),
    .reg_rd_wr_L_in(reg_rd_wr_L_in),
    .reg_addr_in(reg_addr_in),
    .reg_data_in(reg_data_in),
    .reg_src_in(reg_src_in),
    .reg_req_out(reg_req_out),
    .reg_ack_out(reg_ack_out),
    .reg_rd_wr_L_out(reg_rd_wr_L_out),
    .reg_addr_out(reg_addr_out),
    .reg_data_out(reg_data_out),
    .reg_src_out(reg_src_out)
  );

  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  integer cyc;
  initial cyc = 0;
  always @(posedge clk) begin
    cyc <= cyc + 1;
  end

  task reg_write;
    input [7:0] off;
    input [31:0] data;
    begin
      @(posedge clk);
      reg_req_in <= 1'b1;
      reg_ack_in <= 1'b0;
      reg_rd_wr_L_in <= 1'b0;
      reg_addr_in <= A(off);
      reg_data_in <= data;
      reg_src_in <= 2'd0;
      @(posedge clk);
      while (!reg_ack_out) @(posedge clk);
      reg_req_in <= 1'b0;
      reg_ack_in <= 1'b0;
      reg_rd_wr_L_in <= 1'b0;
      reg_addr_in <= 23'd0;
      reg_data_in <= 32'd0;
    end
  endtask

  task reg_read;
    input [7:0] off;
    output [31:0] data;
    begin
      @(posedge clk);
      reg_req_in <= 1'b1;
      reg_ack_in <= 1'b0;
      reg_rd_wr_L_in <= 1'b1;
      reg_addr_in <= A(off);
      reg_data_in <= 32'd0;
      reg_src_in <= 2'd0;
      @(posedge clk);
      while (!reg_ack_out) @(posedge clk);
      data = reg_data_out;
      reg_req_in <= 1'b0;
      reg_ack_in <= 1'b0;
      reg_rd_wr_L_in <= 1'b0;
      reg_addr_in <= 23'd0;
      reg_data_in <= 32'd0;
    end
  endtask

  task imem_wr;
    input [7:0] addr;
    input [31:0] inst;
    begin
      reg_write(OFF_IMEM_ADDR, addr);
      reg_write(OFF_IMEM_WDATA, inst);
      reg_write(OFF_IMEM_WE, 32'd1);
    end
  endtask

  task dmem_wr;
    input [9:0] addr;
    input [63:0] data;
    begin
      reg_write(OFF_DMEM_ADDR, addr);
      reg_write(OFF_DMEM_WLO, data[31:0]);
      reg_write(OFF_DMEM_WHI, data[63:32]);
      reg_write(OFF_DMEM_WE, 32'd1);
    end
  endtask

  task dmem_rd;
    input [9:0] addr;
    output [63:0] data;
    reg [31:0] lo;
    reg [31:0] hi;
    begin
      reg_write(OFF_DMEM_ADDR, addr);
      @(posedge clk);
      reg_read(OFF_DMEM_RLO, lo);
      reg_read(OFF_DMEM_RHI, hi);
      data = {hi, lo};
    end
  endtask

  task set_cfg;
    input [31:0] a;
    input [31:0] b;
    input [31:0] c;
    input [31:0] n;
    begin
      reg_write(OFF_CONTROL, 32'd0);
      reg_write(OFF_A_BASE, a);
      reg_write(OFF_B_BASE, b);
      reg_write(OFF_C_BASE, c);
      reg_write(OFF_N_WORDS, n);
    end
  endtask

  reg trace_en;
  integer trace_start;
  integer trace_limit;

  localparam TRACE_DEPTH = 128;
  integer tptr;

  reg [31:0] tr_cyc   [0:TRACE_DEPTH-1];
  reg [7:0]  tr_pc    [0:TRACE_DEPTH-1];
  reg [31:0] tr_ir    [0:TRACE_DEPTH-1];
  reg [9:0]  tr_tid   [0:TRACE_DEPTH-1];

  reg [9:0]  tr_raddr [0:TRACE_DEPTH-1];
  reg [63:0] tr_rdata [0:TRACE_DEPTH-1];

  reg        tr_we    [0:TRACE_DEPTH-1];
  reg [9:0]  tr_waddr [0:TRACE_DEPTH-1];
  reg [63:0] tr_wdata [0:TRACE_DEPTH-1];

  reg        tr_rfwe  [0:TRACE_DEPTH-1];
  reg [2:0]  tr_rfwad [0:TRACE_DEPTH-1];
  reg [63:0] tr_rfwda [0:TRACE_DEPTH-1];

  reg        tr_ldw   [0:TRACE_DEPTH-1];
  reg        tr_pend  [0:TRACE_DEPTH-1];
  reg [7:0]  tr_pcf   [0:TRACE_DEPTH-1];
  reg [31:0] tr_eff   [0:TRACE_DEPTH-1];

  function [3:0] f_op;
    input [31:0] ir;
    begin
      f_op = ir[31:28];
    end
  endfunction

  task dump_one;
    input integer idx;
    reg [3:0] op;
    begin
      op = f_op(tr_ir[idx]);
      /*$display("cyc=%0d pc=%0d pcf=%0d ir=%h op=%h tid=%0d ldw=%b pend=%b eff=%0d | R[%0d]=%h | WE=%b W[%0d]=%h | RFwe=%b rd=%0d w=%h",
        tr_cyc[idx],
        tr_pc[idx],
        tr_pcf[idx],
        tr_ir[idx],
        op,
        tr_tid[idx],
        tr_ldw[idx],
        tr_pend[idx],
        tr_eff[idx],
        tr_raddr[idx],
        tr_rdata[idx],
        tr_we[idx],
        tr_waddr[idx],
        tr_wdata[idx],
        tr_rfwe[idx],
        tr_rfwad[idx],
        tr_rfwda[idx]
      );*/
    end
  endtask

  task dump_trace;
    integer i;
    integer idx;
    begin
      $display("----- DUMP LAST %0d CYCLES (most recent last) -----", TRACE_DEPTH);
      for (i = 0; i < TRACE_DEPTH; i = i + 1) begin
        idx = (tptr + 1 + i) % TRACE_DEPTH;
        dump_one(idx);
      end
      $display("---------------------------------------------------");
    end
  endtask

  always @(posedge clk) begin
    if (reset) begin
      tptr <= 0;
    end else if (trace_en) begin
      tr_cyc[tptr]   <= cyc;
      tr_pc[tptr]    <= dut.dbg_pc;
      tr_ir[tptr]    <= dut.dbg_ir;
      tr_tid[tptr]   <= dut.dbg_tid;

      tr_raddr[tptr] <= dut.core_dmem_raddr;
      tr_rdata[tptr] <= dut.core_dmem_rdata;

      tr_we[tptr]    <= dut.core_dmem_we;
      tr_waddr[tptr] <= dut.core_dmem_waddr;
      tr_wdata[tptr] <= dut.core_dmem_wdata;

      tr_rfwe[tptr]  <= dut.u_core.rf_we;
      tr_rfwad[tptr] <= dut.u_core.rf_waddr;
      tr_rfwda[tptr] <= dut.u_core.rf_wdata;

      tr_ldw[tptr]   <= dut.u_core.ld_wait;
      tr_pend[tptr]  <= dut.u_core.pending;
      tr_pcf[tptr]   <= dut.u_core.pc_fetch;
      tr_eff[tptr]   <= dut.u_core.eff_addr_word;

      tptr <= (tptr + 1) % TRACE_DEPTH;

      if ((cyc - trace_start) < trace_limit) begin
        /*$display("T=%0t cyc=%0d run=%b pc=%0d pcf=%0d ir=%h tid=%0d ldw=%b pend=%b eff=%0d | R[%0d]=%h | WE=%b W[%0d]=%h | RFwe=%b rd=%0d w=%h",
          $time, cyc, dut.run_reg, dut.dbg_pc, dut.u_core.pc_fetch, dut.dbg_ir, dut.dbg_tid,
          dut.u_core.ld_wait, dut.u_core.pending, dut.u_core.eff_addr_word,
          dut.core_dmem_raddr, dut.core_dmem_rdata,
          dut.core_dmem_we, dut.core_dmem_waddr, dut.core_dmem_wdata,
          dut.u_core.rf_we, dut.u_core.rf_waddr, dut.u_core.rf_wdata
        );*/
      end

      if (dut.dbg_st_we) begin
        /*$display("  ST_EVENT cyc=%0d pc=%0d tid=%0d addr=%0d data=%h",
          cyc, dut.dbg_pc, dut.dbg_tid, dut.dbg_st_addr, dut.dbg_st_data
        );*/
      end
    end
  end

  task run_cycles;
    input integer n;
    integer k;
    begin
      trace_en = 1'b1;
      trace_start = cyc;
      trace_limit = 240;
      reg_write(OFF_CONTROL, 32'd1);
      for (k = 0; k < n; k = k + 1) @(posedge clk);
      reg_write(OFF_CONTROL, 32'd0);
      trace_en = 1'b0;
    end
  endtask

  task pulse_reset;
    begin
      reset = 1'b1;
      trace_en = 1'b0;
      repeat (5) @(posedge clk);
      reset = 1'b0;
      repeat (2) @(posedge clk);
    end
  endtask

  task fill_nops;
    input integer start_pc;
    input integer count;
    integer i;
    begin
      for (i = 0; i < count; i = i + 1) imem_wr(start_pc+i, 32'h00000000);
    end
  endtask

  task load_prog_vadd;
    begin
      imem_wr(8'd0, T(4'h8, 10'd0));
      imem_wr(8'd1, I(4'h2, 3'd1, 3'd0, 3'd0, 2'd0, 16'd0));
      imem_wr(8'd2, I(4'h2, 3'd2, 3'd0, 3'd0, 2'd1, 16'd0));
      imem_wr(8'd3, I(4'h4, 3'd3, 3'd1, 3'd2, 2'd0, 16'd0));
      imem_wr(8'd4, S(4'h3, 3'd3, 2'd2, 16'd0));
      imem_wr(8'd5, I(4'h9, 3'd0, 3'd0, 3'd0, 2'd0, 16'd0));
      imem_wr(8'd6, B(4'hA, 8'd1));
      imem_wr(8'd7, 32'h00000000);
      fill_nops(8, 32);
    end
  endtask

  task load_prog_vsub;
    begin
      imem_wr(8'd0, T(4'h8, 10'd0));
      imem_wr(8'd1, I(4'h2, 3'd1, 3'd0, 3'd0, 2'd0, 16'd0));
      imem_wr(8'd2, I(4'h2, 3'd2, 3'd0, 3'd0, 2'd1, 16'd0));
      imem_wr(8'd3, I(4'h5, 3'd3, 3'd1, 3'd2, 2'd0, 16'd0));
      imem_wr(8'd4, S(4'h3, 3'd3, 2'd2, 16'd0));
      imem_wr(8'd5, I(4'h9, 3'd0, 3'd0, 3'd0, 2'd0, 16'd0));
      imem_wr(8'd6, B(4'hA, 8'd1));
      imem_wr(8'd7, 32'h00000000);
      fill_nops(8, 32);
    end
  endtask

  task load_prog_relu;
    begin
      imem_wr(8'd0, T(4'h8, 10'd0));
      imem_wr(8'd1, I(4'h2, 3'd1, 3'd0, 3'd0, 2'd0, 16'd0));
      imem_wr(8'd2, I(4'h7, 3'd3, 3'd1, 3'd0, 2'd0, 16'd0));
      imem_wr(8'd3, S(4'h3, 3'd3, 2'd2, 16'd0));
      imem_wr(8'd4, I(4'h9, 3'd0, 3'd0, 3'd0, 2'd0, 16'd0));
      imem_wr(8'd5, B(4'hA, 8'd1));
      imem_wr(8'd6, 32'h00000000);
      fill_nops(7, 33);
    end
  endtask

  task load_prog_bf16_mul;
    begin
      imem_wr(8'd0, T(4'h8, 10'd0));
      imem_wr(8'd1, I(4'h2, 3'd1, 3'd0, 3'd0, 2'd0, 16'd0));
      imem_wr(8'd2, I(4'h2, 3'd2, 3'd0, 3'd0, 2'd1, 16'd0));
      imem_wr(8'd3, I(4'hB, 3'd3, 3'd1, 3'd2, 2'd0, 16'd0));
      imem_wr(8'd4, S(4'h3, 3'd3, 2'd2, 16'd0));
      imem_wr(8'd5, I(4'h9, 3'd0, 3'd0, 3'd0, 2'd0, 16'd0));
      imem_wr(8'd6, B(4'hA, 8'd1));
      imem_wr(8'd7, 32'h00000000);
      fill_nops(8, 32);
    end
  endtask

  task load_prog_bf16_fma;
    begin
      imem_wr(8'd0, T(4'h8, 10'd0));
      imem_wr(8'd1, I(4'h2, 3'd1, 3'd0, 3'd0, 2'd0, 16'd0));
      imem_wr(8'd2, I(4'h2, 3'd2, 3'd0, 3'd0, 2'd1, 16'd0));
      imem_wr(8'd3, I(4'h2, 3'd4, 3'd0, 3'd0, 2'd2, 16'd0));
      imem_wr(8'd4, I(4'hC, 3'd3, 3'd1, 3'd2, 2'd0, 16'h0004));
      imem_wr(8'd5, S(4'h3, 3'd3, 2'd2, 16'd0));
      imem_wr(8'd6, I(4'h9, 3'd0, 3'd0, 3'd0, 2'd0, 16'd0));
      imem_wr(8'd7, B(4'hA, 8'd1));
      imem_wr(8'd8, 32'h00000000);
      fill_nops(9, 31);
    end
  endtask

  task init_dmem_vadd;
    begin
      dmem_wr(10'd0, 64'h0004_0003_0002_0001);
      dmem_wr(10'd1, 64'h0008_0007_0006_0005);
      dmem_wr(10'd16, 64'h000D_000C_000B_000A);
      dmem_wr(10'd17, 64'h0003_0000_FFFF_FFFE);
      dmem_wr(10'd32, 64'h0000_0000_0000_0000);
      dmem_wr(10'd33, 64'h0000_0000_0000_0000);
    end
  endtask

  task init_dmem_vsub;
    begin
      dmem_wr(10'd0, 64'h0004_0003_0002_0001);
      dmem_wr(10'd1, 64'h0008_0007_0006_0005);
      dmem_wr(10'd16, 64'h000F_000E_000D_000C);
      dmem_wr(10'd17, 64'h0007_0006_0005_0004);
      dmem_wr(10'd32, 64'h0000_0000_0000_0000);
      dmem_wr(10'd33, 64'h0000_0000_0000_0000);
    end
  endtask

  task init_dmem_relu;
    begin
      dmem_wr(10'd0, 64'h0002_0001_8001_0004);
      dmem_wr(10'd1, 64'h8003_8004_8005_8006);
      dmem_wr(10'd32, 64'h0);
      dmem_wr(10'd33, 64'h0);
    end
  endtask

  task init_dmem_bf16;
    begin
      dmem_wr(10'd0, 64'h4080_4040_4000_3F80);
      dmem_wr(10'd1, 64'h4000_3F80_4080_3F80);
      dmem_wr(10'd16, 64'hBF80_3F80_3F00_4000);
      dmem_wr(10'd17, 64'h3F80_3F80_BF80_3F80);
      dmem_wr(10'd32, 64'h3F80_3F80_3F80_3F80);
      dmem_wr(10'd33, 64'h3FC0_3F80_40A0_3FC0);
    end
  endtask

  reg [63:0] r0;
  reg [63:0] r1;

  task check;
    input [63:0] got;
    input [63:0] exp;
    input [255:0] msg;
    begin
      if (got !== exp) begin
        $display("[FAIL] %s got=%h exp=%h", msg, got, exp);
        dump_trace();
        $finish;
      end else begin
        $display("[PASS] %s = %h", msg, got);
      end
    end
  endtask

  initial begin
    reg_req_in = 0;
    reg_ack_in = 0;
    reg_rd_wr_L_in = 0;
    reg_addr_in = 0;
    reg_data_in = 0;
    reg_src_in = 0;
    trace_en = 0;
    trace_start = 0;
    trace_limit = 240;
    tptr = 0;

    pulse_reset();
    set_cfg(32'd0, 32'd16, 32'd32, 32'd2);

    $display("=== TEST1: i16 vadd ===");
    init_dmem_vadd();
    load_prog_vadd();
    run_cycles(140);
    dmem_rd(10'd32, r0);
    dmem_rd(10'd33, r1);
    check(r0, 64'h0011_000F_000D_000B, "vadd.t0");
    check(r1, 64'h000B_0007_0005_0003, "vadd.t1");

    pulse_reset();
    set_cfg(32'd0, 32'd16, 32'd32, 32'd2);

    $display("=== TEST2: i16 vsub ===");
    init_dmem_vsub();
    load_prog_vsub();
    run_cycles(140);
    dmem_rd(10'd32, r0);
    dmem_rd(10'd33, r1);
    check(r0, 64'hFFF5_FFF5_FFF5_FFF5, "vsub.t0");
    check(r1, 64'h0001_0001_0001_0001, "vsub.t1");

    pulse_reset();
    set_cfg(32'd0, 32'd16, 32'd32, 32'd2);

    $display("=== TEST3: i16 relu ===");
    init_dmem_relu();
    load_prog_relu();
    run_cycles(140);
    dmem_rd(10'd32, r0);
    dmem_rd(10'd33, r1);
    check(r0, 64'h0002_0001_0000_0004, "relu.t0");
    check(r1, 64'h0000_0000_0000_0000, "relu.t1");

    pulse_reset();
    set_cfg(32'd0, 32'd16, 32'd32, 32'd2);

    $display("=== TEST4: bf16 vmul ===");
    init_dmem_bf16();
    load_prog_bf16_mul();
    run_cycles(220);
    dmem_rd(10'd32, r0);
    dmem_rd(10'd33, r1);
    check(r0, 64'hC080_4040_3F80_4000, "bf16_mul.t0");
    check(r1, 64'h4000_3F80_C080_3F80, "bf16_mul.t1");

    pulse_reset();
    set_cfg(32'd0, 32'd16, 32'd32, 32'd2);

    $display("=== TEST5: bf16 vfma ===");
    init_dmem_bf16();
    load_prog_bf16_fma();
    run_cycles(260);
    dmem_rd(10'd32, r0);
    dmem_rd(10'd33, r1);
    check(r0, 64'hC040_4080_4000_4040, "bf16_fma.t0");
    check(r1, 64'h4060_4000_3F80_4020, "bf16_fma.t1");

    $display("All tests passed.");
    $finish;
  end

endmodule
